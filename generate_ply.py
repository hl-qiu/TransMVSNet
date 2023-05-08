import argparse

import argon2

from utils import *
from datasets.data_io import read_pfm, save_pfm
from plyfile import PlyData, PlyElement
from PIL import Image
import os

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
# pair文件所在目录
parser.add_argument('--testpath', default='data/DTU/mvs_testing/dtu', help='testing data dir for some scenes')
# 原先scan的结果所在目录（因为需要原先的深度图、相机参数）
parser.add_argument('--outdir', default='outputs/dtu_testing', help='output dir')
# scan列表文件
parser.add_argument('--testlist', default='lists/dtu/test1.txt', help='testing scene list')
parser.add_argument('--out_folder', default='outputs/opaque_output', help='testing scene list')
# partial_mask文件所在目录
parser.add_argument('--partial_mask_dir', default='data/opaque/opaque_2', help='testing scene list')

parser.add_argument('--num_view', type=int, default=5, help='num of view')
parser.add_argument('--h', type=int, default=800, help='testing max h')
parser.add_argument('--w', type=int, default=800, help='testing max w')
# filter
parser.add_argument('--conf', type=float, default=0.03, help='prob confidence')
parser.add_argument('--thres_view', type=int, default=5, help='threshold of num view')

# 参数解析和检查
args = parser.parse_args()


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsics, extrinsics


# read an images
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    # TODO step1. 将ref视图像素 投影 到src视图下 project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))  # 参考视图的坐标
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])  # 拉成一维
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]  # 归一化坐标z

    # TODO step2. 重投影reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    # 重投影回来的坐标是世界坐标系下的坐标，第三维z就是深度值
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)  # 重投影回来的像素坐标（未归一化）
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]  # 归一化坐标，得到像素坐标
    # 重投影回来的像素坐标
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    #
    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


# 几何一致性检查
def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    # 得到几何掩码
    mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0  # 将那些不满足条件的位置的深度置为0

    # 返回几何一致性约束的掩码图、经过掩码图过滤的重投影深度图、src视图下的像素坐标x2d_src和y2d_src
    return mask, depth_reprojected, x2d_src, y2d_src


# 深度图过滤
def filter_depth(pair_folder, scan_folder, txtfilename, plyfilename, scanName):
    # 配对数据
    pair_data = read_pair_file(os.path.join(pair_folder, "pair.txt"))
    # 读取图片id列表
    with open(os.path.join(args.outdir, "{}/img_list.txt".format(scanName)), 'r') as f:
        imgList = list(map(int, f.readline().split(',')))
    img_dict = {}
    for i, img in enumerate(imgList):
        img_dict[i] = img

    vertexs = []  # 最终的点云
    vertex_colors = []  # 点云的颜色
    # 遍历pair文件中的每一组(ref_view，[src_views])
    for ref_view, src_views in pair_data:
        # if ref_view not in imgList:
        #     continue
        # src_views = src_views[:args.num_view]  # 控制src视图的个数
        # 加载ref相机参数
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # 加载ref图片
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # 加载ref的深度图
        ref_depth_est = read_pfm(os.path.join(scan_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]

        # 将所有掩码合并起来
        geo_mask_sum = 0
        all_srcview_depth_ests = []  # 所有src视图的深度图
        for src_view in src_views:
            # if src_view not in imgList:
            #     continue
            # 加载src视图的相机参数
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # 加载src视图的深度图
            src_depth_est = read_pfm(os.path.join(scan_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(
                ref_depth_est, ref_intrinsics, ref_extrinsics, src_depth_est, src_intrinsics, src_extrinsics)

            geo_mask_sum += geo_mask.astype(np.int32)  # 累加几何掩码图
            all_srcview_depth_ests.append(depth_reprojected)  # 收集src深度图

        # 平均深度图估计
        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # TODO 读取final_mask
        img = Image.open(os.path.join(scan_folder, os.path.join('mask', '{:0>8}_final.png'.format(ref_view))))
        final_mask = np.array(img, dtype=np.int32).reshape((args.h, args.w))
        # TODO 读取partial_mask
        # imgList = [11, 13, 15, 19, 2, 28, 31, 33, 34, 36, 46, 47, 53, 58, 61, 64, 68, 69, 72, 73, 79, 8, 80, 83, 84, 85, 95, 97, 99]

        opaqueName = args.partial_mask_dir.split('/')[-1]
        partial_mask = np.load(
            os.path.join(args.partial_mask_dir, 'opaque_{}_{}.npy'.format(ref_view, int(opaqueName[-1])))
        ).reshape((args.h, args.w))
        # TODO 取交
        final_mask = np.logical_and(partial_mask, final_mask)
        # cv2.imwrite('bbbb.jpg', final_mask* 255)

        # 生成点云
        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = final_mask
        print("valid_points", valid_points.mean())

        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        # color = ref_img[1:-16:4, 1::4, :][valid_points]  # hardcoded for DTU dataset

        # 获得有效点的颜色
        color = ref_img[valid_points]
        # 得到相机坐标系下的坐标
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        # 得到世界坐标系下的坐标
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))
    # 拼接起来
    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    # 将每一行转换为一个tuple，每一个tuple代表一个点，由(x,y,z)坐标表示
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    # TODO 将点云写成txt格式文件(不带颜色)
    np.savetxt(txtfilename, vertexs)

    # TODO 将点云写成ply格式文件(带颜色)
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
      # 保存点云ply
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def generate(scan):
    scan_id = int(scan[4:])
    save_name = 'mvsnet{:0>3}_partial.ply'.format(scan_id)

    pair_folder = os.path.join(args.testpath, scan)  # pair文件所在目录
    scan_folder = os.path.join(args.outdir, scan)  # 原先scan的结果所在目录（因为需要原先的深度图、相机参数）
    # 生成点云
    out_dir = os.path.join(args.out_folder, 'opaque_{}'.format(int(args.partial_mask_dir.split('/')[-1][-1])))     # 输出结果的保存目录
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 保存点云txt的位置
    txtfilename = os.path.join(out_dir, 'points.txt')
    # 保存点云ply的位置
    plyfilename = os.path.join(out_dir, save_name)
    
    # 深度图滤波
    filter_depth(pair_folder, scan_folder, txtfilename, plyfilename, scan)


if __name__ == '__main__':
    # 读取场景数据目录名称
    with open(args.testlist) as f:
        content = f.readlines()
        testlist = [line.rstrip() for line in content]
    for scan in testlist:
        generate(scan)
