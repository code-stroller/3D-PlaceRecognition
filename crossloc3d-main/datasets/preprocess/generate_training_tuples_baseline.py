# generate_training_tuples_refine.py
# ──────────────────────────────────────────────────────────────────────────────
# PointNetVLAD 방식(KDTree)으로 “inhouse + oxford” Refine 학습 쿼리를 생성합니다.
#
# 사용 예시:
#   cd <CrossLoc3D 루트>/datasets/preprocess/
#   python generate_training_tuples_refine.py --dataset_root ../    \
#       --output training_queries_refine_v2.pickle
# ──────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
import argparse
from sklearn.neighbors import KDTree
import pickle
import random

# -----------------------------------------------------------------------------
# 1) 테스트 영역 경계 정의 (P1~P10)  → Oxford 원본 기준
# -----------------------------------------------------------------------------
P1 = [5735712.768124, 620084.402381]
P2 = [5735611.299219, 620540.270327]
P3 = [5735237.358209, 620543.094379]
P4 = [5734749.303802, 619932.693364]
P5 = [5734639.900457, 619082.463074]
P6 = [5734379.377172, 619113.618144]
P7 = [5734278.080852, 619052.400099]
P8 = [5734012.669263, 618832.031873]
P9 = [5733896.840572, 619206.929244]
P10 = [5733660.243307, 618806.788889]
TEST_POINTS = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10]

# test 여부를 판정할 x/y 범위 (PointNetVLAD 원본: 150m)
X_WIDTH = 150
Y_WIDTH = 150

def check_in_test_set(northing, easting, points, x_width, y_width):
    """
    주어진 (northing, easting)가 points 리스트 중
    어떤 중심점의 ±(x_width, y_width) 범위 내에 있으면 True.
    """
    for (px, py) in points:
        if (px - x_width < northing < px + x_width) and \
           (py - y_width < easting < py + y_width):
            return True
    return False


# -----------------------------------------------------------------------------
# 2) KDTree 기반으로 positives/negatives 인덱스 생성 함수
# -----------------------------------------------------------------------------
def construct_query_dict(df_centroids, save_path, radius_pos=10, radius_neg=50):
    """
    df_centroids: columns=['file','northing','easting'] DataFrame
    save_path: pickle 파일로 저장될 경로
    radius_pos: (m) 내에서 이웃들을 긍정 샘플로 지정
    radius_neg: (m) 바깥에서 이웃들을 부정 샘플로 지정

    최종 생성되는 포맷:
       queries = {
         0: {'query': '.../path/to/file0.bin', 'positives': [2,10,5,...], 'negatives':[1,3,4,...]},
         1: {...}, 
         ...
       }
    """
    coords = df_centroids[['northing','easting']].to_numpy()
    tree = KDTree(coords)

    # radius 기준으로 index 리스트 추출
    ind_nn = tree.query_radius(coords, r=radius_pos)  # radius_pos 내 모든 인덱스 (자기 자신 포함)
    ind_r  = tree.query_radius(coords, r=radius_neg)  # radius_neg 내 모든 인덱스

    queries = {}
    all_indices = np.arange(len(df_centroids))

    for i in range(len(df_centroids)):
        query_path = df_centroids.iloc[i]['file']  # 파일 경로 (문자열)
        # radius_pos 내에서 i 자신만 제외 → positives
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()
        # radius_neg 내에 있는 모든 점 인덱스를 제외한 나머지 → negatives
        negatives = np.setdiff1d(all_indices, ind_r[i]).tolist()
        random.shuffle(negatives)

        queries[i] = {
            'query': query_path,
            'positives': positives,
            'negatives': negatives
        }

    # pickle로 저장
    with open(save_path, 'wb') as f:
        pickle.dump(queries, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Done] Saved {len(queries)} queries → {save_path}")


# -----------------------------------------------------------------------------
# 3) 메인 로직: Inhouse + Oxford 데이터 읽어서 train set 구성 후 pickle 생성
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Refined training tuples for Inhouse+Oxford'
    )
    parser.add_argument(
        '--dataset_root', 
        type=str, 
        required=True,
        help='데이터셋 루트 경로 (예: data/benchmark_datasets/)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='training_queries_refine.pickle',
        help='생성될 pickle 파일명 (default: training_queries_refine.pickle)'
    )
    args = parser.parse_args()

    # 예: args.dataset_root 가 “../data/benchmark_datasets/”와 같은 경로가 된다.
    base_path = args.dataset_root.rstrip('/') + '/'

    # ====================================================================
    # 3-1) Inhouse 데이터 처리
    #    - inhouse_datasets/<run>/pointcloud_centroids_10.csv
    #    - inhouse_datasets/<run>/pointcloud_25m_10/<timestamp>.bin
    # ====================================================================
    df_train = pd.DataFrame(columns=['file','northing','easting'])
    inhouse_runs_folder = "inhouse_datasets/"
    inhouse_csv_name   = "pointcloud_centroids_10.csv"
    inhouse_pcd_folder = "/pointcloud_25m_10/"

    # os.listdir 결과를 정렬한 뒤 모든 sub-folder 순회
    inhouse_folders = sorted(os.listdir(os.path.join(base_path, inhouse_runs_folder)))
    for folder in inhouse_folders:
        csv_path = os.path.join(base_path, inhouse_runs_folder, folder, inhouse_csv_name)
        df_loc = pd.read_csv(csv_path, sep=',')
        # “inhouse_runs_folder + folder + inhouse_pcd_folder + timestamp + .bin”
        df_loc['file'] = (
            inhouse_runs_folder + folder 
            + inhouse_pcd_folder 
            + df_loc['timestamp'].astype(str)
            + '.bin'
        )
        # test 영역에 속하면 제외
        mask = df_loc.apply(
            lambda row: not check_in_test_set(
                row['northing'], row['easting'],
                TEST_POINTS, X_WIDTH, Y_WIDTH
            ), 
            axis=1
        )
        df_valid = df_loc.loc[mask, ['file','northing','easting']].reset_index(drop=True)
        df_train = pd.concat([df_train, df_valid], ignore_index=True)

    # ====================================================================
    # 3-2) Oxford 데이터 처리
    #    - oxford/<run>/pointcloud_locations_20m_10overlap.csv
    #    - oxford/<run>/pointcloud_20m_10overlap/<timestamp>.bin
    #    (마지막 폴더(테스트 전용)는 사용하지 않음)
    # ====================================================================
    oxford_runs_folder = "oxford/"
    oxford_csv_name    = "pointcloud_locations_20m_10overlap.csv"
    oxford_pcd_folder  = "/pointcloud_20m_10overlap/"

    oxford_folders = sorted(os.listdir(os.path.join(base_path, oxford_runs_folder)))
    # 보통 마지막 폴더(oxford[-1])는 테스트 전용이므로, 그 이전까지 사용
    for folder in oxford_folders[:-1]:
        csv_path = os.path.join(base_path, oxford_runs_folder, folder, oxford_csv_name)
        df_loc = pd.read_csv(csv_path, sep=',')
        df_loc['file'] = (
            oxford_runs_folder + folder 
            + oxford_pcd_folder 
            + df_loc['timestamp'].astype(str)
            + '.bin'
        )
        mask = df_loc.apply(
            lambda row: not check_in_test_set(
                row['northing'], row['easting'],
                TEST_POINTS, X_WIDTH, Y_WIDTH
            ), 
            axis=1
        )
        df_valid = df_loc.loc[mask, ['file','northing','easting']].reset_index(drop=True)
        df_train = pd.concat([df_train, df_valid], ignore_index=True)

    # ====================================================================
    # 3-3) 결과 확인, 인덱스 리셋 및 KDTree→pickle 생성
    # ====================================================================
    df_train = df_train.reset_index(drop=True)
    print(f"Total training submaps (Inhouse+Oxford, excl. test): {len(df_train)}")

    save_pickle = os.path.join(base_path, args.output)
    construct_query_dict(df_train, save_pickle)

    print("All done. Refine queries saved to:", save_pickle)
