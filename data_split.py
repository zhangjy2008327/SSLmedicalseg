import os
import cv2
import csv
import math
import random
import shutil


def move(filepath, destpath):
    # 从数据集中随机选取20%移动到另一文件夹下作为测试集，剩下的80%作为训练集
    #这里我们需要将数据集划分成训练集为0.7 测试集为0.3
    pathdir = os.listdir(filepath)
    ranpath = random.sample(pathdir, int(0.1 * len(pathdir)))
    print(len(pathdir))
    print(len(ranpath))
    print(ranpath)
    for alldir in ranpath:
        child = os.path.join(filepath, alldir)
        dest = os.path.join(destpath, alldir)
        if not alldir.startswith('.'):#隐藏文件不参与划分
            print(child)
            shutil.copytree(child, dest)#文件夹
            shutil.rmtree(child)#文件夹
            # shutil.copy(child, dest) #文件
            # os.remove(child) #文件


def move_label(imgpath, labelpath, testpath):
    # 根据不同文件夹下的图片移动相应图片的标签
    labels = os.listdir(labelpath)
    for label in labels:
        imgdir = os.listdir(imgpath)
        for img in imgdir:
            if label.strip('.txt') == img.strip('.jpg'):
                print('###')
                label_path = os.path.join(labelpath, label)
                test_path = os.path.join(testpath, label)
                print(label_path)
                if labelpath:
                    shutil.copy(label_path, test_path)
                    os.remove(label_path)


def move_img(imgpath, labelpath, testpath):
    labels = os.listdir(labelpath)
    for label in labels:
        imgdir = os.listdir(imgpath)
        for img in imgdir:
            if label.strip('.txt') == img.strip('.jpg'):
                print('###')
                img_path = os.path.join(imgpath, img)
                test_path = os.path.join(testpath, img)
                shutil.copy(img_path, test_path)
                os.remove(img_path)


if __name__ == "__main__":
    move('E:\dataset\MICCAI_BraTS_2018_Data\pretrain', 'E:\\dataset\\MICCAI_BraTS_2018_Data\\fine-tuning')

    # move_img("./train", './labels/test_labels', './test')
