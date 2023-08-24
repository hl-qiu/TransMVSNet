import os
import shutil

import cv2
import numpy as np
from PIL import Image

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
    directory = "/home/ubuntu/qhl/TransMVSNet/data/DTU/mvs_testing/dtu/scan33/images"
    for i in range(49):
        # j = 8361 + i
        # j = i % 7
        # shutil.copyfile(os.path.join(directory, 'IMG_' + str(j) + '.txt'), os.path.join(directory, '%08d_cam.txt' % i))
        shutil.copyfile(os.path.join(directory, 'rect_{:0>3}_3_r5000.png'.format(i+1)), os.path.join(directory, '%08d.jpg' % i))
        # shutil.copyfile(os.path.join(directory, '%08d.txt' % i), os.path.join(directory, '%08d_cam.txt' % i))
    # 将png转为jpg
    # directory1 = "/home/ubuntu/qhl/TransMVSNet/data/DTU/mvs_testing/dtu/scan0425/images1"
    # for i in range(100):
        # with Image.open(os.path.join(directory1, '%08d.png' % i)) as img:
            # # 转换为RGB模式
            # img = img.convert('RGB')
            # # 保存为JPG图像
            # img.save(os.path.join("/data/DTU/mvs_testing/dtu/scan0425/images", '%08d.jpg' % i))
    

    #
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
    # partial_mask = np.load('/home/ubuntu/qhl/TransMVSNet/data/opaque/ship/opaque_ship_2/opaque_0_2.npy')
    # partial_mask2 = np.load('/home/ubuntu/qhl/TransMVSNet/data/opaque/opaque_scan33_2/opaque_6_2.npy')
    # partial_mask4 = np.load('/home/ubuntu/qhl/TransMVSNet/data/opaque/opaque_scan62_0/opaque_6_0.npy')
    # partial_mask3= partial_mask.reshape((800, 800)).astype(np.uint8)
    # cv2.imwrite('aaaaaaa20.jpg', partial_mask*255)

    # 读取ply点云
    # out_arr = read_ply_cloud('/home/ubuntu/qhl/TransMVSNet/outputs2/mvsnet000_partial3.ply')
    # out_arr = out_arr[:1004083]
    # np.savetxt('/home/ubuntu/qhl/TransMVSNet/banana2.txt', out_arr)
    # print("output array from input list : ", out_arr)
    # print("shape : ", out_arr.shape)





