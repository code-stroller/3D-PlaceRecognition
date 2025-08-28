####PointNetVLAD Datasets####
-All submaps are in binary file format
-Ground truth GPS coordinate of the submaps are found in the csv files
-Use CSV files to define positive and negative point clouds
-Filename of the submaps are their timestamps which is consistent with the timestamps in the csv files
-All submaps are preprocess with the road removed and downsampled to 4096 points


##Oxford

-45 sets in total of full and partial runs
-Used both full and partial runs for training
-Only used full runs for testing/inference

-Training submaps found in the folder "pointcloud_20m_10overlap/"
-GPS coordinate of each submap can be found in "pointcloud_locations_20m_10overlap.csv"
-Training submaps are not mutually disjoint per run
-Each submap ~20m of car trajectory and subsequent submaps are ~10m apart

-Test/Inference submaps found in the folder "pointcloud_20m/"
-GPS coordinate of each submap can be found in "pointcloud_locations_20m"
-Test/Inference submaps are mutually disjoint


##Inhouse Datasets

-Each inhouse dataset has 5 runs
-Training submaps are found in the folder "pointcloud_25m_10/" and its corresponding csv file is "pointcloud_centroids_10.csv"
-Test/Infenrence submaps are found in the folder "pointcloud_25m_25/" and its corresponding csv file is "pointcloud_centroids_25.csv"