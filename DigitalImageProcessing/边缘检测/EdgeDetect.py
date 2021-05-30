"""
SJTU-AU333-数字图像处理-作业2边缘检测
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import os
from Smoothing import medianSmooth

# 支持中文标题
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置

# 输入输出目录与算子列表
srcpath = './MSGaussRes'
respath = './EdgeDetectRes'
bipath = './BiRes'
opeLst = ['sobel', 'lap', 'LoG', 'canny']


def edgeDetect(filename, ope):
    """
    使用不同的算子进行边缘检测，包括sobel、Lapalcian、高斯-Laplacian（Marr算子）、Canny算子
    算子尺寸都选择3
    :param img: 图片数据，numpy,mat
    :param ope: 使用的算子，可选sobel, lap, LoG, canny
    :return: 边缘矩阵、算子名称
    """
    if ope not in opeLst:
        raise ValueError('算子参数错误，尝试使用\'sobel\', \'lap\', \'LoG\', \'canny\'')

    # 读取文件
    file = os.path.join(srcpath, filename)
    filehead = os.path.splitext(filename)[0]
    filetail = os.path.splitext(filename)[1]
    img = cv2.imread(file)

    res = None
    if ope == 'sobel':
        dx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
        dy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
        dx = cv2.convertScaleAbs(dx)
        dy = cv2.convertScaleAbs(dy)
        res = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
    elif ope == 'lap':
        res = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
        res = cv2.convertScaleAbs(res)
    elif ope == 'LoG':
        res = cv2.GaussianBlur(img, (3, 3), 0)
        res = cv2.Laplacian(res, cv2.CV_16S, ksize=3)
        res = cv2.convertScaleAbs(res)
    elif ope == 'canny':
        res = cv2.Canny(img, 40, 110)

    outfilehead = filehead + '-' + ope
    cv2.imshow(outfilehead, res)
    cv2.waitKey(100)
    cv2.imwrite(os.path.join(respath, outfilehead + filetail), res)

    return res, ope


def edgeDetectNoWrite(img, ope):
    """
    使用不同的算子进行边缘检测，包括sobel、Lapalcian、高斯-Laplacian（Marr算子）、Canny算子
    算子尺寸都选择3
    :param img: 图片数据，numpy,mat
    :param ope: 使用的算子，可选sobel, lap, LoG, canny
    :return: 边缘矩阵、算子名称
    """
    if ope not in opeLst:
        raise ValueError('算子参数错误，尝试使用\'sobel\', \'lap\', \'LoG\', \'canny\'')

    res = None
    if ope == 'sobel':
        dx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
        dy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
        dx = cv2.convertScaleAbs(dx)
        dy = cv2.convertScaleAbs(dy)
        res = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
    elif ope == 'lap':
        res = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
        res = cv2.convertScaleAbs(res)
    elif ope == 'LoG':
        res = cv2.GaussianBlur(img, (3, 3), 0)
        res = cv2.Laplacian(res, cv2.CV_16S, ksize=3)
        res = cv2.convertScaleAbs(res)
    elif ope == 'canny':
        res = cv2.Canny(img, 40, 110)

    return res


def binaryzation(filename, threshold):
    """
    使用给定的阈值进行二值化
    :param img: 图像数据
    :param threshold: 阈值
    :return: 二值化图像，与原图像同尺寸
    """
    # 读取文件
    file = os.path.join(respath, filename)
    img = cv2.imread(file)

    biGph = img.copy()
    biGph[biGph >= threshold] = 255
    biGph[biGph < threshold] = 0

    cv2.imshow(filename, biGph)
    cv2.waitKey(100)
    cv2.imwrite(os.path.join(bipath, filename), biGph)

    return biGph


if __name__ == '__main__':
    # print('Hello world')
    #
    # # 所有算子测试
    # for ope in opeLst:
    #     for filename in os.listdir(srcpath):
    #         res, op = edgeDetect(filename, ope)
    # cv2.destroyAllWindows()
    #
    # # 所有算子测试结果二值化
    # for ope in opeLst:
    #     for filename in os.listdir(respath):
    #         bires = binaryzation(filename, 30)
    # cv2.destroyAllWindows()

    # 在sobel、Canny算子中选用不同的梯度阈值测试
    fname = os.path.join(respath, 'gn(5, 5)-sobel.jpg')
    img = cv2.imread(fname)
    thLst = [40, 70, 100, 130, 160]
    lowThLst = [10, 40, 70, 40, 40]
    highLst = [110, 110, 110, 80, 140]
    # # sobel不同阈值
    # for th in thLst:
    #     tmp = img.copy()
    #     tmp[tmp >= th] = 255
    #     tmp[tmp < th] = 0
    #     cv2.imshow(fname, tmp)
    #     cv2.waitKey(0)
    #     cv2.imwrite('./MThRes/' + str(th) + '-gn(5, 5)-sobel.jpg', tmp)
    # canny不同阈值
    # fname = os.path.join(srcpath, 'gn(5, 5).jpg')
    # img = cv2.imread(fname)
    # for i in range(5):
    #     low = lowThLst[i]
    #     high = highLst[i]
    #     res = cv2.Canny(img, low, high)
    #     cv2.imshow(fname, res)
    #     cv2.waitKey(0)
    #     cv2.imwrite('./MThRes/' + str(low) + '-' + str(high) + '-gn(5, 5)-sobel.jpg', res)

    # 噪声影响测试
    f1 = './EdgeDetect/noise-no.bmp'
    f2 = './EdgeDetect/noise-salt.jpg'
    img1 = cv2.imread(f1)
    img2 = cv2.imread(f2)
    # res1 = edgeDetectNoWrite(img1, 'sobel')
    # res2 = edgeDetectNoWrite(img1, 'lap')
    # res3 = edgeDetectNoWrite(img2, 'sobel')
    # res4 = edgeDetectNoWrite(img2, 'lap')
    # cv2.imwrite('./NoiseRes/' + 'sobel-' + 'noise-no.bmp', res1)
    # cv2.imwrite('./NoiseRes/' + 'lap-' + 'noise-no.bmp', res2)
    # cv2.imwrite('./NoiseRes/' + 'sobel-' + 'noise-salt.bmp', res3)
    # cv2.imwrite('./NoiseRes/' + 'lap-' + 'noise-salt.bmp', res4)

    # 中值平滑+canny解决椒盐噪声
    res5 = medianSmooth(img2, (7, 7))
    res5 = edgeDetectNoWrite(res5, 'canny')
    cv2.imwrite('./NoiseRes/' + 'median+canny-' + 'noise-salt.bmp', res5)
