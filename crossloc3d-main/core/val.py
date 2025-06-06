import torch

from datasets import create_dataloaders
from tasks import create_task
from .eval import eval


def val(cfg, log, task=None):

    if task is None:
        task = create_task(cfg.task_type, cfg, log)
        if torch.cuda.is_available():
            task.cuda()
        assert cfg.resume_from is not None
        task_state = task.load(cfg.resume_from)

    if isinstance(cfg.dataset_type, (list, tuple)):
        dataset_types = cfg.dataset_type
    else:
        dataset_types = [cfg.dataset_type]

    metrics_all = []
    for dt in dataset_types:
        log.info('++++++++ Evaluating %s dataset ++++++++' % dt)
        (db_dl, _), (q_dl, _) = create_dataloaders(
            dataset_type=dt, cfg=cfg, subset_types=('database','queries'), log=log, debug=cfg.debug)
        m = eval(cfg, log, db_dl, q_dl, task)
        metrics_all.append(m)
    return metrics_all
