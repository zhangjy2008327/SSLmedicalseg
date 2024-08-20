import glob
import multiprocessing
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np

from joblib import Parallel, delayed


def parallel_load_brats_no_labels(path):
    # flair_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith("_flair.nii.gz")])
    flair_files = sorted(glob.glob(path + "*_flair.nii.gz", recursive=True))
    # print(flair_files)
    print("loading " + str(len(flair_files)) + " mri scans.")

    num_cores = multiprocessing.cpu_count()# cpu.coount()返回当前计算机上的cpu核心数
    all_slices = list()#创建空列表

    results = Parallel(n_jobs=num_cores)(
        delayed(read_brats_scan_multimodal)(flair_files, i,) for i in
        range(len(flair_files)))

    for mm_scan in results:
        num = 0
        # print(mm_scan.shape)
        # print(mm_scan)
        # exit(0)
        for z in range(0, mm_scan.shape[2], 7):#遍历每个模态图像数据数组的切片
           if num < 20:#每隔七个取一次切片
                num += 1
                flair_image = mm_scan[:, :, z]
                all_slices.append(flair_image)
           else :
               break
    return np.array(all_slices)


def parallel_load_brats_with_labels(path): #这个可以用于验证和测试部分  分别返回的是切片和mask
    ma_x = 0
    flair_files = sorted(glob.glob(path + "*_flair.nii.gz", recursive=True))
    seg_files = sorted(glob.glob(path + "*_seg.nii.gz", recursive=True))
    # flair_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith("_flair.nii.gz")])
    # seg_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith("_seg.nii.gz")])
    print("loading " + str(len(seg_files)) + " mri scans.")

    num_cores = multiprocessing.cpu_count()
    all_slices = list()
    all_masks = list()

    results = Parallel(n_jobs=num_cores)(
        delayed(read_brats_scan_multimodal_with_mask)(flair_files, i, seg_files)
        for i in range(len(flair_files)))
    for mm_scan in results:
        num = 0

        for z in range(0, mm_scan[0].shape[2], 7):  # 遍历每个模态图像数据数组的切片
            if (num < 20) :
            # 每隔七个取一次切片
                num += 1
                flair_image = mm_scan[0][:, :, z]
                ma_x = max(ma_x, np.max(flair_image))

                # print(flair_image.shape)
                seg_image = mm_scan[1][:, :, z]


                all_slices.append(flair_image)
                all_masks.append(seg_image)
    print("ma_x-------",ma_x)
    return np.array(all_slices), np.array(all_masks)



def read_brats_scan_multimodal(flair_files, i):#用于读取多模态的BRATS数据集中的MRI图像 并对其进行预处理
    new_size = (240, 240, 155)
    flair_image, nbbox = read_scan_find_bbox(nib.load(flair_files[i]))#每一维全空的区域剔除掉
    flair_image_tensor = torch.from_numpy(flair_image).unsqueeze(0).unsqueeze(0)
    # 使用插值方法进行大小调整
    flair_image_resized = F.interpolate(flair_image_tensor, size=new_size, mode='trilinear', align_corners=False)
    # 将新像素点限制在指定范围内
    # flair_image_resized = torch.clamp(flair_image_resized, 0, 1)
    # 删除批次维度和通道维度，并获取调整后的 3D 图像数据
    flair_image = flair_image_resized.squeeze(0).squeeze(0)
    flair_image = np.asarray(flair_image, dtype=np.float32)

    return flair_image


def read_brats_scan_multimodal_with_mask(flair_files, i ,seg_files):
    new_size = (240, 240, 155)
    flair_image, nbbox = read_scan_find_bbox(nib.load(flair_files[i]))
    # flair_image = skTrans.resize(flair_image, new_resolution, order=1, preserve_range=True)
    # 将 NumPy 数组转换为 PyTorch 张量，并添加批次维度
    flair_image_tensor = torch.from_numpy(flair_image).unsqueeze(0).unsqueeze(0)
    # print(flair_image_tensor)
    # print(flair_image_tensor.shape)
    # exit(0)
    # 使用插值方法进行大小调整
    flair_image_resized = F.interpolate(flair_image_tensor, size=new_size, mode='trilinear', align_corners=False)
    # 删除批次维度和通道维度，并获取调整后的 3D 图像数据
    flair_image = flair_image_resized.squeeze(0).squeeze(0)
    flair_image = np.asarray(flair_image, dtype=np.float32)

    seg_image = read_scan(nbbox, nib.load(seg_files[i]), normalize=False)
    seg_image_tensor = torch.from_numpy(seg_image).unsqueeze(0).unsqueeze(0)
    # 使用插值方法进行大小调整
    seg_image_resized = F.interpolate(seg_image_tensor, size=new_size, mode='trilinear', align_corners=False)
    # 删除批次维度和通道维度，并获取调整后的 3D 图像数据
    seg_image = seg_image_resized.squeeze(0).squeeze(0)


    seg_image = np.asarray(seg_image, dtype=np.int32)
    seg_image[seg_image == 4] = 3#这里进行了类别合并  将4这个类别合并到3中
    seg_image[seg_image == 3] = 0#把ET归为背景类  change
    seg_image[seg_image == 2] = 0#把TC归为背景类  change
    return flair_image, seg_image



def read_scan_find_bbox(nif_file, normalize=True):#这个normalize用于指定是否对图像数据进行归一化

    ans = 0
    image = nif_file.get_fdata()#get_fdata()用于获取图像的原始数据 具体说：image是一个多维numpy数组，包含了图像的像素值



    # for i in image:
    #     for j in i:
    #         for k in j:
    #
    #             if k!=0 :
    #                 ans+=1
    #
    # print(ans)
    # exit(0)
    # all_pixels = np.all(image == 0)
    # if all_pixels:
    #     count += 1
    #     print("wy")
    # print("pppppppp")
    # print(image)
    # exit(0)
    st_x, en_x, st_y, en_y, st_z, en_z = 0, 0, 0, 0, 0, 0#定义的6个变量  用于记录图像在x、y、z方向上的边界范围
    #以下的几个循环都是为了通过找到非空切片来确定边界值
    # print(image.shape)
    # print("image")
    # print(image)
    for x in range(image.shape[0]):#如果image 是一个三维数据  那么image.shape返回一个元组 (height,width,depth)
        # 其中height表示图像在垂直方向上的像素数量 表示为image.shape[0]
        if np.any(image[x, :, :]):#用于检查给定切片image[x,:,:]是否存在非零（空）的像素值 若存在返回True  否则此切片便是全空的
            st_x = x
            # print("str----")
            # print(st_x)
            break
    for x in range(image.shape[0] - 1, -1, -1):#倒序遍历 从image.shape[0]开始，-1位置结束，-1是步长
        if np.any(image[x, :, :]):
            en_x = x
            # print("end_x----")
            # print(en_x)
            break
    for y in range(image.shape[1]):
        if np.any(image[:, y, :]):
            st_y = y
            break
    for y in range(image.shape[1] - 1, -1, -1):
        if np.any(image[:, y, :]):
            en_y = y
            break
    for z in range(image.shape[2]):
        if np.any(image[:, :, z]):
            st_z = z
            break
    for z in range(image.shape[2] - 1, -1, -1):
        if np.any(image[:, :, z]):
            en_z = z
            break
    image = image[st_x:en_x, st_y:en_y, st_z:en_z]#去除图像中的空白区域 只保留有值的部分

    if normalize:
        image = norm(image)
    nbbox = np.array([st_x, en_x, st_y, en_y, st_z, en_z]).astype(int)
    return image, nbbox


def read_scan(sbbox, nif_file, normalize=True):
    if normalize:
        image = norm(nif_file.get_fdata()[sbbox[0]:sbbox[1], sbbox[2]:sbbox[3], sbbox[4]:sbbox[5]])
    else:
        image = nif_file.get_fdata()[sbbox[0]:sbbox[1], sbbox[2]:sbbox[3], sbbox[4]:sbbox[5]]
    return image


def norm(im):
    im = im.astype(np.float32)
    min_v = np.min(im)
    max_v = np.max(im)
    im = (im - min_v) / (max_v - min_v)
    return im



if __name__ == "__main__":
    count = 0
    brats_path = "E:/MICCAI_BraTS_2018_Data_Training/HGG/**/"

    with_masks = True


    if with_masks:
        brats_X, brats_Y = parallel_load_brats_with_labels(brats_path)
        # print(brats_X[0])
        # n = 0
        # for i in brats_X:
        #     all_pixels = np.all(i == 0)
        #     if(all_pixels):
        #         n += 1
        #         print(i)
        #
        # print("-------",n)

        # print("IMAGE")
        # print(brats_X[9])
        # print("MASK")
        # print(brats_Y[9])
        print(brats_X.shape, brats_Y.shape)
        print("max--x",np.max(brats_X[10]))
        print("max--y", np.max(brats_Y[10]))
        # print(count)

        # np_to_tfrecords_with_labels(brats_X, brats_Y, file_prefix, verbose=True, multimodal=multimodal)
    else:
        brats_data = parallel_load_brats_no_labels(brats_path)

        # print(len(brats_data))
        # print("222222")
        print(brats_data.shape)

        # np_to_tfrecords_no_labels(brats_data, file_prefix, verbose=True, multimodal=multimodal)

#切片出来的都是0值
#对3D图像进行大小变换的时候出现错误  flair_image = np.asarray(flair_image, dtype=np.float32)  本来这里写的是np.int32

