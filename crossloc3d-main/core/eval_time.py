import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KDTree
import time
from utils import AverageValue, Metrics


def get_recall(m, n, database_vectors, query_vectors, query_sets,
               num_neighbors=25, db_set=None):
    # Original PointNetVLAD recall routine (변경 없음)
    if db_set is None:
        pair_dist = None
    else:
        db_set = db_set[m]
        pair_dist = []

    database_output = database_vectors[m]
    queries_output = query_vectors[n]
    database_nbrs = KDTree(database_output)

    recall = [0] * num_neighbors
    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        dist = []
        query_details = query_sets[n][i]
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            dist.append(np.linalg.norm(
                np.array(db_set[indices[0][j]]) -
                np.array([query_details["northing"],
                          query_details["easting"]]), ord=1))
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break
        if len(list(set(indices[0][:threshold])
                    .intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

        pair_dist.append(min(dist))

    if num_evaluated == 0:
        return None, None, None, None

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100
    return recall, top1_similarity_score, one_percent_recall, pair_dist


def _measure_forward(task, meta, data, latencies_ms):
    """
    Forward-pass latency 측정 (ms/scan).
    CUDA 환경은 torch.cuda.Event, CPU 환경은 time.perf_counter 사용.
    """
    batch_size = len(meta['idx'])

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        embeddings = task.step(meta, data)
        end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end)          # ms (batch)
    else:
        t0 = time.perf_counter()
        embeddings = task.step(meta, data)
        latency = (time.perf_counter() - t0) * 1e3 # ms (batch)

    latencies_ms.append(latency / batch_size)      # ms/scan
    return embeddings


def eval(cfg, log, db_data_loader, q_data_loader, task, neighbor=25):

    metrics = AverageValue(Metrics.names())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    task.eval()

    # ────────── 추론 지연 기록용 리스트 ──────────
    latencies_ms = []

    # ------------------- Database pass ------------------- #
    all_db_embs, all_db_set = [], []
    for i in tqdm(range(db_data_loader.dataset.subset_len())):
        db_data_loader.dataset.set_subset(i)
        db_embeddings, db_set = [], []

        for meta, data in db_data_loader:
            with torch.no_grad():
                embeddings = _measure_forward(task, meta, data, latencies_ms)

            if cfg.eval_cfg.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            db_embeddings.append(embeddings.detach().cpu().numpy())

            for idx in range(len(meta['idx'])):
                db_set.append((meta["northing"][idx], meta["easting"][idx]))
            if cfg.debug:
                break

        db_embeddings = np.concatenate(db_embeddings, axis=0)
        all_db_embs.append(db_embeddings)
        all_db_set.append(np.array(db_set))

    # ------------------- Query pass ------------------- #
    all_q_embs = []
    for i in tqdm(range(q_data_loader.dataset.subset_len())):
        q_data_loader.dataset.set_subset(i)
        q_embeddings = []

        for meta, data in q_data_loader:
            with torch.no_grad():
                embeddings = _measure_forward(task, meta, data, latencies_ms)

            if cfg.eval_cfg.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            q_embeddings.append(embeddings.detach().cpu().numpy())
            if cfg.debug:
                break

        q_embeddings = np.concatenate(q_embeddings, axis=0)
        all_q_embs.append(q_embeddings)

    # ------------------- Recall 계산 ------------------- #
    iters = [(i, j) for i in range(len(all_db_embs))
                     for j in range(len(all_q_embs)) if i != j]

    similarity, dist = [], []
    recall = np.zeros(neighbor)

    with tqdm(total=len(iters)) as pbar:
        for i, j in iters:
            pair_recall, pair_similarity, pair_opr, pair_dist = get_recall(
                i, j, all_db_embs, all_q_embs,
                q_data_loader.dataset.catalog,
                num_neighbors=neighbor, db_set=all_db_set)

            if pair_recall is None:
                continue
            metrics.update(Metrics.get(pair_opr))
            recall += np.array(pair_recall)
            similarity.extend(pair_similarity)
            if pair_dist is not None:
                dist.extend(pair_dist)
            pbar.update(1)

    avg_recall     = recall / len(iters)
    avg_similarity = float(np.mean(similarity))
    avg_dist       = float(np.mean(dist))

    # ------------------- 결과 로깅 ------------------- #
    mean_lat = np.mean(latencies_ms)
    min_lat  = np.min(latencies_ms)
    max_lat  = np.max(latencies_ms)

    log.info('====================== EVALUATE RESULTS ======================')

    format_str = '{sample_num:<10} ' + \
                 ' '.join(['{%s:<10}' % vn for vn in metrics.value_names])

    header_dict = dict(sample_num='Sample')
    header_dict.update({vn: vn for vn in metrics.value_names})
    log.info(format_str.format(**header_dict))

    result_dict = dict(sample_num=len(iters))
    result_dict.update({vn: "%.4f" % metrics.avg(vn)
                        for vn in metrics.value_names})
    log.info(format_str.format(**result_dict))

    t = ('Avg. similarity: {:.4f}  Avg. dist: {:.4f}  '
         'Avg. recall @N:\n{}').format(avg_similarity, avg_dist, avg_recall)
    log.info(t)

    log.info('Inference latency per scan [ms] → '
             f'mean: {mean_lat:.3f}   min: {min_lat:.3f}   max: {max_lat:.3f}')

    # 리턴 형식은 원본과 동일
    return Metrics('Recall@1%', metrics.avg())
