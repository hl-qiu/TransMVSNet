import numpy as np

edit = 3
file = f'E:\myProject\pcl\ship\out_point_clouds.txt'
# out_file = f"./data/out_point_clouds_correct_{edit}.txt"
out_file = f'E:\myProject\pcl\ship\out_point_clouds_correct.txt'
point = np.loadtxt(file)
# point[point[...,3]==-1] = 3
t = np.bincount(point[..., 3].astype(np.int32))
sort = np.argsort(t)[-4:]
print(t)
print(sort)
point_cloud = []
for i in sort:
    point_cloud.append(point[point[..., 3] == i])

point_cloud = np.concatenate(point_cloud, axis=0)
np.savetxt(out_file, point_cloud)
