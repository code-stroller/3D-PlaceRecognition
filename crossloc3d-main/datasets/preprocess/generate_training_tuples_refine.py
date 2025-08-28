# generate_training_tuples_refine.py
# ──────────────────────────────────────────────────────────────────────────────
# “Inhouse + Oxford” Refine 학습용 KDTree 쿼리(pickle) 생성 스크립트
# 
# - argparse 없이 바로 실행해도 동작하도록 구현
# - CrossLoc3D 기준으로, 스크립트 위치(datasets/preprocess/)에서 자동으로
#   프로젝트 루트 → data/benchmark_datasets/ 폴더를 찾아 들어감
# - 최종 결과: data/benchmark_datasets/training_queries_refine_v2.pickle
# ──────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import pickle
import random
import sys

# -----------------------------------------------------------------------------
# 1) “inhouse + oxford” 데이터셋이 들어 있는 루트 경로 자동 탐색
#
#    스크립트 위치: <프로젝트_루트>/datasets/preprocess/generate_training_tuples_refine.py
#    → 프로젝트 루트: ../..  → 그 아래 data/benchmark_datasets/ 폴더를 가정
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
DEFAULT_BASE_PATH = os.path.join(PROJECT_ROOT, 'data', 'benchmark_datasets')

if not os.path.isdir(DEFAULT_BASE_PATH):
    sys.stderr.write(f"[Error] 기대한 데이터 루트가 없습니다:\n  {DEFAULT_BASE_PATH}\n")
    sys.exit(1)

# 이 스크립트를 수정 없이 바로 실행하면, 아래 경로에 pickle이 생성됩니다.
OUTPUT_FILENAME = 'training_queries_refine_v2.pickle'
OUTPUT_PATH = os.path.join(DEFAULT_BASE_PATH, OUTPUT_FILENAME)


# -----------------------------------------------------------------------------
# 2) Oxford 원본의 “Test 영역” 좌표 10개 (PointNetVLAD 기준)
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

# “Test 영역” 판정용 half-width/half-height (meters)
X_WIDTH = 150
Y_WIDTH = 150

def check_in_test_set(northing, easting, points, x_width, y_width):
    """
    (northing, easting)가 TEST_POINTS 중 하나의 박스(±x_width, ±y_width) 안에 들어가면 True.
    """
    for (px, py) in points:
        if (px - x_width < northing < px + x_width) and (py - y_width < easting < py + y_width):
            return True
    return False


# -----------------------------------------------------------------------------
# 3) KDTree를 이용해 positives/negatives 인덱스 미리 계산 → pickle 저장
# -----------------------------------------------------------------------------
def construct_query_dict(df_centroids, save_path, radius_pos=10, radius_neg=50):
    """
    df_centroids: ['file','northing','easting'] 컬럼 DataFrame
    save_path: pickle로 저장할 경로 (문자열)
    radius_pos: (m) 이내 이웃 인덱스 → positives
    radius_neg: (m) 이내 이웃 인덱스 제외 → negatives
    """
    coords = df_centroids[['northing','easting']].to_numpy()
    tree = KDTree(coords)

    # radius_pos, radius_neg 기준으로 index array 생성
    ind_nn = tree.query_radius(coords, r=radius_pos)  # (len, 변수 길이)
    ind_r  = tree.query_radius(coords, r=radius_neg)  # (len, 변수 길이)

    total_n = len(df_centroids)
    queries = {}
    all_indices = np.arange(total_n)

    for i in range(total_n):
        query_filepath = df_centroids.iloc[i]['file']
        # radius_pos 이내에서 자기자신 빼고 → positives
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()
        # radius_neg 이내 인덱스 전부 제외하고 → negatives
        negatives = np.setdiff1d(all_indices, ind_r[i]).tolist()
        random.shuffle(negatives)

        queries[i] = {
            'query':      query_filepath,
            'positives':  positives,
            'negatives':  negatives
        }

    # pickle 저장
    with open(save_path, 'wb') as f:
        pickle.dump(queries, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Done] Saved {len(queries)} queries → {save_path}")


# -----------------------------------------------------------------------------
# 4) 메인 로직: Inhouse + Oxford 데이터를 하나의 DataFrame으로 합친 뒤 KDTree 쿼리 생성
# -----------------------------------------------------------------------------
def main():
    base_path = DEFAULT_BASE_PATH

    # (1) Inhouse 데이터 읽기
    df_train = pd.DataFrame(columns=['file','northing','easting'])
    inhouse_runs_folder = "inhouse_datasets/"
    inhouse_csv_name   = "pointcloud_centroids_10.csv"
    inhouse_pcd_folder = "/pointcloud_25m_10/"

    inhouse_folder_list = sorted(os.listdir(os.path.join(base_path, inhouse_runs_folder)))
    for folder in inhouse_folder_list:
        csv_path = os.path.join(base_path, inhouse_runs_folder, folder, inhouse_csv_name)
        df_loc  = pd.read_csv(csv_path, sep=',')
        # “inhouse_datasets/<run>/pointcloud_centroids_10.csv” → file 컬럼에 “inhouse_datasets/<run>/pointcloud_25m_10/<timestamp>.bin” 추가
        df_loc['file'] = (
            inhouse_runs_folder + folder
            + inhouse_pcd_folder
            + df_loc['timestamp'].astype(str)
            + '.bin'
        )
        # TEST 영역(Olford – 10개 박스) 에 속하는 샘플은 제외
        mask_ok = df_loc.apply(
            lambda row: not check_in_test_set(
                row['northing'], row['easting'], TEST_POINTS, X_WIDTH, Y_WIDTH
            ),
            axis=1
        )
        df_valid = df_loc.loc[mask_ok, ['file','northing','easting']].reset_index(drop=True)
        df_train = pd.concat([df_train, df_valid], ignore_index=True)

    # (2) Oxford 데이터 읽기
    oxford_runs_folder = "oxford/"
    oxford_csv_name    = "pointcloud_locations_20m_10overlap.csv"
    oxford_pcd_folder  = "/pointcloud_20m_10overlap/"

    oxford_folder_list = sorted(os.listdir(os.path.join(base_path, oxford_runs_folder)))
    # “마지막” 폴더는 보통 자체 테스트 전용이므로, 그 이전까지만 사용
    for folder in oxford_folder_list[:-1]:
        csv_path = os.path.join(base_path, oxford_runs_folder, folder, oxford_csv_name)
        df_loc  = pd.read_csv(csv_path, sep=',')
        df_loc['file'] = (
            oxford_runs_folder + folder
            + oxford_pcd_folder
            + df_loc['timestamp'].astype(str)
            + '.bin'
        )
        mask_ok = df_loc.apply(
            lambda row: not check_in_test_set(
                row['northing'], row['easting'], TEST_POINTS, X_WIDTH, Y_WIDTH
            ),
            axis=1
        )
        df_valid = df_loc.loc[mask_ok, ['file','northing','easting']].reset_index(drop=True)
        df_train = pd.concat([df_train, df_valid], ignore_index=True)

    # (3) 최종 DataFrame 리셋 및 출력
    df_train = df_train.reset_index(drop=True)
    print(f"Total training submaps (Inhouse + Oxford, excl. test): {len(df_train)}")

    # (4) pickle 생성
    construct_query_dict(df_train, OUTPUT_PATH)
    print("All done. Pickle file created at:", OUTPUT_PATH)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
