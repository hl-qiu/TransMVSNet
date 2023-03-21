import os
import shutil

import cv2
import numpy as np
from datasets.data_io import read_pfm
import numpy as np
from plyfile import PlyData

def read_ply_cloud(filename):
    ply_data = PlyData.read(filename)
    points = ply_data['vertex'].data.copy()
    print(points.shape)
    cloud = np.empty([6513005, 3])
    for i in range(len(points)):
        point = points[i]
        p = np.array([point[0], point[1], point[2]])
        cloud[i] = p
    return np.array(cloud)

if __name__ == '__main__':
    # 修改image图片的名称
    # directory = "/home/ubuntu/qhl/TransMVSNet/data/DTU/scan00/cams"
    # for i in range(107):
    #     # j = 8361 + i
    #     # shutil.copyfile(os.path.join(directory, 'IMG_' + str(j) + '.txt'), os.path.join(directory, '%08d_cam.txt' % i))
    #     shutil.copyfile(os.path.join(directory, '%08d.txt' % i), os.path.join(directory, '%08d_cam.txt' % i))

    # 读取pfm深度图文件
    # a = read_pfm('/home/ubuntu/qhl/TransMVSNet/outputs/dtu_testing/scan9/depth_est/00000000.pfm')
    # print(a)

    # 将cam文件中的内参除以4
    # for i in range(116):
    #     lists = []
    #     with open("/home/ubuntu/qhl/mvsNet_pytorch/data/scan4/cams/%08d_cam.txt" % i, 'r') as f:
    #         lines = f.readlines()
    #         for j in range(len(lines)):
    #             if j == 7 or j == 8:
    #                 line = ' '.join([str(float(tmp)/4) for tmp in lines[j].split()])
    #                 lists.append(line + '\n')
    #             else:
    #                 lists.append(lines[j])
    #     with open("/home/ubuntu/qhl/mvsNet_pytorch/data/scan4/cams/%08d_cam.txt" % i, "w") as f:
    #         for line in lists:
    #             f.write(line)

    # imgList = np.loadtxt('/home/ubuntu/qhl/TransMVSNet/outputs/dtu_testing/scan00/img_list.txt')
    # with open('/home/ubuntu/qhl/TransMVSNet/outputs/dtu_testing/scan00/img_list.txt', 'r') as f:
    #     imgList = f.readline().split(',')
    #     imgList = list(map(int, imgList))
    #     print(imgList)
    # TODO 读取partial_mask
    # partial_mask = np.load('/home/ubuntu/qhl/TransMVSNet/outputs/dtu_testing/scan00/opaque_out/opaque_20_0.npy').reshape((1504, 2016)).astype(np.uint8)
    # cv2.imwrite('aaaaaaa20.jpg', partial_mask*255)

    # 读取ply点云
    out_arr = read_ply_cloud('/home/ubuntu/qhl/TransMVSNet/outputs2/mvsnet000_partial3.ply')
    out_arr = out_arr[:1004083]
    np.savetxt('/home/ubuntu/qhl/TransMVSNet/banana2.txt', out_arr)
    print("output array from input list : ", out_arr)
    print("shape : ", out_arr.shape)



