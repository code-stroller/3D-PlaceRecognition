from tqdm import tqdm
import torch
import os.path as osp
from time import time

from datasets import create_dataloaders
from tasks import create_task
from optimizers import create_optimizer
from schedulers import create_scheduler
from utils import AverageValue, Metrics
from .val import val


def train(cfg, log):

    train_data_loader, train_batch_sampler = create_dataloaders(
        dataset_type=cfg.dataset_type,
        cfg=cfg,
        subset_types='train',
        log=log,
        debug=cfg.debug
    )

    task = create_task(cfg.task_type, cfg, log)

    if torch.cuda.is_available():
        task.cuda()

    if cfg.resume_from is not None:
        task_state = task.load(cfg.resume_from)
        start_epoch = task_state['epoch'] if 'epoch' in cfg.resume_items else 0
        optimizer = create_optimizer(
            cfg.optimizer_type, cfg.optimizer_cfg, task.model_params())
        scheduler = create_scheduler(
            cfg.scheduler_type, cfg.scheduler_cfg, optimizer)
        if 'optim' in cfg.resume_items:
            optimizer.load_state_dict(task_state['optim_state_dict'])
        if 'sched' in cfg.resume_items:
            scheduler.load_state_dict(task_state['sched_state_dict'])
        if 'sampler' in cfg.resume_items and hasattr(train_batch_sampler, 'load_state_dict'):
            train_batch_sampler.load_state_dict(
                task_state['sampler_state_dict'])
        best_metrics = Metrics(
            'Recall@1%', task_state['best_metrics']) if 'metrics' in cfg.resume_items else None
    else:
        start_epoch = 0
        optimizer = create_optimizer(
            cfg.optimizer_type, cfg.optimizer_cfg, task.model_params())
        scheduler = create_scheduler(
            cfg.scheduler_type, cfg.scheduler_cfg, optimizer)
        best_metrics = None

    end_epoch = cfg.end_epoch

    for epoch in range(start_epoch + 1, end_epoch + 1):

        log.info('[Epoch%4d/%4d] Start training ...' % (epoch, end_epoch))
        epoch_start_time = time()
        losses = None

        task.before_epoch(epoch)
        task.train()

        with tqdm(total=len(train_data_loader)) as pbar:
            for batch_idx, (meta, data) in enumerate(train_data_loader):

                optimizer.zero_grad()
                loss_info = task.step(meta, data)
                if losses is None:
                    losses = AverageValue(list(loss_info.keys()))
                losses.update(loss_info)
                optimizer.step()
                torch.cuda.empty_cache()
                details = {}
                details.update(losses.avg() if type(losses.avg())
                               == dict else {'loss': losses.avg()})
                pbar.set_postfix(**details)
                pbar.update(1)

                if cfg.debug:
                    break
        scheduler.step()

        epoch_end_time = time()

        log.info(
            '[Epoch%4d/%4d] Training time= %.3fs %s' %
            (epoch, end_epoch, epoch_end_time - epoch_start_time, losses.avg_str()))

        save_dir = osp.join(cfg.work_dir, 'ckpt_tmp.pth')
        task.save(
            save_dir=save_dir,
            epoch=epoch,
            best_metrics=None,
            optim_state_dict=optimizer.state_dict(),
            sched_state_dict=scheduler.state_dict(),
            sampler_state_dict=train_batch_sampler.state_dict() if hasattr(
                train_batch_sampler, 'state_dict') else None
        )

        if epoch % cfg.train_cfg.val_per_epoch == 0 or epoch == end_epoch:
            log.info('[Epoch%4d/%4d] Start validating ...' %
                     (epoch, end_epoch))
            epoch_start_time = time()
            # --- 원래는 val()이 리스트를 반환하므로, 첫 번째 Metrics 객체만 사용하도록 수정 ---
            metrics_list = val(cfg, log, task)
            metrics = metrics_list[0]
            epoch_end_time = time()
            log.info(
                '[Epoch%4d/%4d] Validating time= %.3fs' %
                (epoch, end_epoch, epoch_end_time - epoch_start_time))

            # ─── “오직 Oxford Recall@1%”로 best checkpoint 고르기 ────────────────────
            # 1) metrics 안에서 Oxford/Recall@1% 값을 꺼냅니다.
            #    (아래는 state_dict()에서 직접 꺼내는 예시입니다.)
            curr_oxford_recall = metrics.state_dict().get('oxford/Recall@1%', None)

            if best_metrics is None:
                better = True
            else:
                prev_oxford_recall = best_metrics.state_dict().get('oxford/Recall@1%', None)
                # “이전보다 현재 값이 크면” True
                better = (curr_oxford_recall is not None and
                          prev_oxford_recall is not None and
                          curr_oxford_recall > prev_oxford_recall)

            if better:
                best_metrics = metrics
                save_dir = osp.join(cfg.work_dir, 'best_ckpt.pth')
                task.save(
                    save_dir=save_dir,
                    epoch=epoch,
                    best_metrics=metrics.state_dict(),
                    optim_state_dict=optimizer.state_dict(),
                    sched_state_dict=scheduler.state_dict(),
                    sampler_state_dict=train_batch_sampler.state_dict() if hasattr(
                        train_batch_sampler, 'state_dict') else None
                )

            if epoch % cfg.train_cfg.save_per_epoch == 0 or epoch == end_epoch:
                save_dir = osp.join(cfg.work_dir, 'ckpt[epoch=%d].pth' % epoch)
                task.save(
                    save_dir=save_dir,
                    epoch=epoch,
                    best_metrics=metrics.state_dict(),
                    optim_state_dict=optimizer.state_dict(),
                    sched_state_dict=scheduler.state_dict(),
                    sampler_state_dict=train_batch_sampler.state_dict() if hasattr(
                        train_batch_sampler, 'state_dict') else None
                )
        if train_batch_sampler is not None:
            train_batch_sampler.update(losses.avg())

        task.after_epoch(epoch)

        if cfg.debug:
            break
