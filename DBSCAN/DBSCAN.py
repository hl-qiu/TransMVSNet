# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

import open3d as o3d
import numpy as np

if __name__ == '__main__':
    edit = 3
    # file_path = f'../data/fruit/opaque_3/opaque_{edit}.txt'
    file_path = '/home/ubuntu/qhl/TransMVSNet/outputs/opaque_output/scan33/opaque_2/points.txt'
    file_out_path = f'/home/ubuntu/qhl/TransMVSNet/outputs/opaque_output/scan33/opaque_2/out_point_clouds.txt'

    points = np.loadtxt(file_path, delimiter=' ')[::2]
    # 将点云转换为 Open3D 中的 PointCloud 类型
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # pcd = o3d.io.read_point_cloud(file_path, format="xyz")
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 指定显示为灰色
    # labels返回聚类成功的类别，-1表示没有分到任何类中的点
    print(pcd)
    labels = np.array(pcd.cluster_dbscan(eps=0.20, min_points=30, print_progress=True))
    # 最大值+1相当于共有多少个类别
    max_label = np.max(labels)
    print(max(labels))
    # 共有n+1个类别的颜色，n表示聚类成功的类别，1表示没有分类成功的类别
    colors = np.random.randint(255, size=(max_label + 1, 3)) / 255.
    colors = colors[labels]     # 给点标上颜色
    # 没有分类成功的点设置为黑色
    colors[labels < 0] = 0
    labels[labels < 0] = max_label + 1
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 点云显示
    o3d.visualization.draw_geometries([pcd],  # 点云列表
                                      window_name="DBSCAN聚类",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度
    # points = np.concatenate((pcd.points,labels),-1)
    labels = np.reshape(labels, (-1, 1))
    points = np.concatenate((np.asarray(pcd.points), labels), -1)
    print(points)
    np.savetxt(file_out_path, points)
