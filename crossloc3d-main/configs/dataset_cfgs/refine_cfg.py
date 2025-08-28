# configs/dataset_cfgs/refine_cfg.py
# ──────────────────────────────────────────────────────────────────────────────────
# Inhouse + Oxford 합친 Refine용 학습 쿼리 구성 (생성 스크립트: generate_training_tuples_refine.py)
dataset_type = 'Oxford'  # 실제 데이터셋 클래스는 OxfordDataset을 그대로 재사용
data_root_dir = 'data/benchmark_datasets/'

dataset_cfg = dict(
    data_root_dir = data_root_dir,
    transform_cfg = dict(
        train = [
            dict(type='Jitter3D', objs=['pcd'], params=dict(sigma=0.001, clip=0.002)),
            dict(type='Drop3D', objs=['pcd'], params=dict(method='random', min_dr=0.0, max_dr=0.1)),
            dict(type='Translation3D', objs=['pcd'], params=dict(max_delta=0.01)),
            dict(type='Drop3D', objs=['pcd'], params=dict(method='cuboid', p=0.4, bbox_pc_name='pcd')),
            dict(type='ToTensor', objs=['pcd'], params=dict()),
        ],
        database = [
            dict(type='ToTensor', objs=['pcd'], params=dict()),
        ],
        queries = [
            dict(type='ToTensor', objs=['pcd'], params=dict()),
        ],
    ),
    batch_transform_cfg = dict(
        train = [
            dict(type='Rotation3D', objs=['pcd'], params=dict(batch=True, max_theta1=5, max_theta2=0, axis=[0,0,1])),
            dict(type='Mirror3D', objs=['pcd'], params=dict(batch=True, method='xyz_plane', p=[0.25,0.25,0.])),
        ],
        database = [],
        queries  = [],
    ),

    # ───────────────────────────────────────────────────────────────────────────────
    # 학습용 catalog
    train_catalog_file_path = data_root_dir + 'training_queries_refine.pickle',
    # 한 번만 만들어두면 다음부터는 cached 사용
    cached_train_catalog_file_path = data_root_dir + 'training_queries_refine_cached.pickle',

    # 평가 데이터(특별히 변경 없이 Oxford만 사용)
    database_file_path = data_root_dir + 'oxford_evaluation_database.pickle',
    queries_file_path  = data_root_dir + 'oxford_evaluation_query.pickle',
)
# ──────────────────────────────────────────────────────────────────────────────────
