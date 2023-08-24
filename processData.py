import numpy as np
import os

# 定义缩放因子和降采样因子
scale_factor = 1.0 / 200
# scale_factor = 1.0
downsample = 4.0

# 获取文件夹路径
folder_path = "/home/ubuntu/data/scan33/Cameras"

# 遍历文件夹中的所有txt文件
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)

        # 读取txt文件内容
        with open(file_path, 'r') as file:
            content = file.readlines()

        # 解析外参矩阵
        extrinsic = []
        for line in content[1:5]:
            extrinsic.append(list(map(float, line.strip().split())))
        extrinsic = np.array(extrinsic)

        # 解析内参矩阵
        intrinsic = []
        for line in content[7:10]:
            intrinsic.append(list(map(float, line.strip().split())))
        intrinsic = np.array(intrinsic)

        # 解析深度范围
        depth_range = np.array(list(map(float, content[11].strip().split())))

        # 进行缩放和降采样操作
        extrinsic[:3, 3] *= scale_factor
        intrinsic[:2, :] *= downsample
        depth_range[0:] *= scale_factor

        # 创建新目录
        new_folder_path = "/home/ubuntu/qhl/TransMVSNet/data/DTU/mvs_testing/dtu/scan33/cams"
        os.makedirs(new_folder_path, exist_ok=True)

        # 写入新文件
        new_file_path = os.path.join(new_folder_path, filename)
        with open(new_file_path, 'w') as new_file:
            for i, line in enumerate(content):
                if i == 0 or i == 5 or i == 6 or i == 10:
                    new_file.write(line)
                elif i >= 1 and i <= 4:
                    line = ' '.join(str(val) for val in extrinsic[i - 1]) + '\n'
                    new_file.write(line)
                elif i >= 7 and i <= 9:
                    line = ' '.join(str(val) for val in intrinsic[i - 7]) + '\n'
                    new_file.write(line)
                elif i == 11:
                    line = ' '.join(str(val) for val in depth_range)
                    new_file.write(line)

        print(f"已成功将修改后的信息保存至 {new_file_path} 文件中。")
