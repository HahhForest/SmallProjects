"""
SJTU-AU338-数字图像处理-作业1图像平滑
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import *

# 支持中文标题
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置


def rgb2gray(rgb):
    """
    彩色图像转为灰度图像，使用人眼的敏感度
    :return: 灰度图像，类型为num.mat
    """
    print('灰度化...')
    gray = mat(zeros((rgb.shape[0], rgb.shape[1]), dtype=uint8))
    # 已经是灰度图
    if size(rgb.shape) == 2:
        return rgb
    # 单通道灰度图
    if rgb.shape[2] == 1:
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                gray[i, j] = rgb[i, j, 0]

    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            gray[i, j] = 0.11 * rgb[i, j, 0] + 0.59 * rgb[i, j, 1] + 0.3 * rgb[i, j, 2]
    return gray


def meanValueSmooth(img, kSize = (5, 5)):
    """
    均值平滑：取卷积模板内所有像素的均值为当前像素值。使用黑像素填充图像周围
    :param img: 图像数据，mat
    :param kSize: 卷积模板尺寸，正奇数
    :return: 平滑后的图像
    """
    print('平均平滑...')
    normWeight = kSize[0] * kSize[1]
    anchor = (int((kSize[0] - 1) / 2), int((kSize[1] - 1) / 2))
    tmplt = mat(np.ones((kSize[0], kSize[1]), dtype=float32)) / normWeight
    return cv2.filter2D(img, cv2.CV_8U, tmplt, anchor=anchor, borderType=cv2.BORDER_ISOLATED)


def gaussianSmooth(img, kSize=(5, 5)):
    """
    高斯滤波，调用cv2.GaussianBlur()实现
    :param img: 原图像
    :param kSize: 高斯核尺寸，width和height必须为正奇数
    :return: 滤波后的图像
    """
    print('高斯平滑...')
    res = mat(zeros((img.shape[0], img.shape[1]), dtype=uint8))
    cv2.GaussianBlur(img, ksize=kSize, sigmaX=0, dst=res)
    return res


def medianSmooth(img, kSize=(5, 5)):
    """
    中值滤波，取窗口内的中值为当前像素值
    :param img: 原图像
    :param kSize: 窗口尺寸，最好为正奇数
    :return: 平滑后的图像
    """
    print('中值平滑...')
    # 窗口的半部，简化后面运算
    littleHalf0 = (kSize[0] - 1) // 2
    littleHalf1 = (kSize[1] - 1) // 2
    greatHalf0 = kSize[0] - littleHalf0
    greatHalf1 = kSize[1] - littleHalf1
    # 首先图像扩充
    expansioned = cv2.copyMakeBorder(img, littleHalf0, littleHalf0,
                                     littleHalf1, littleHalf1,
                                     borderType=cv2.BORDER_ISOLATED)
    print('扩充前尺寸-' + str(img.shape) + ', 扩充后尺寸-' + str(expansioned.shape))

    # 计算
    dst = mat(np.zeros((expansioned.shape[0], expansioned.shape[1]), dtype=uint8))
    for i in range(littleHalf0, expansioned.shape[0]-littleHalf0):
        for j in range(littleHalf1, expansioned.shape[1]-littleHalf1):
            dst[i, j] = np.round(np.median(expansioned[i-littleHalf0: i+greatHalf0, j-littleHalf1:j+greatHalf1], ))

    return dst[littleHalf0: expansioned.shape[0]-littleHalf0, littleHalf1: expansioned.shape[1]-littleHalf1]


def constructGaussKernel(kSize=5, sigma=0):
    """
    构造高斯滤波卷积核，这里考虑简单情况，卷积核必须为方形
    若sigma为0，则使用kSize构造，认为μ+3σ外影响很小
    :param kSize:卷积核尺寸
    :param sigma:标准差。若sigma为0，则使用kSize构造，认为μ+3σ外影响很小，因此简单构造delta = (kSize+2)/6
    :return:
    """
    if kSize % 2 != 1:
        raise ValueError('卷积核尺寸应为正奇数')
    if sigma == 0:
        sigma = (kSize + 2) / 6
    print('计算高斯核...尺寸-' + str(kSize) + ', sigma-' + str(sigma))

    # 计算。前面的常系数省略不算，最后归一化即可
    sigma2 = sigma**2
    kernel = mat(np.zeros((kSize, kSize), dtype=float64))
    anchor = (kSize - 1) // 2
    total = 0
    for i in range(kSize):
        for j in range(kSize):
            kernel[i, j] = np.math.exp(-((i - anchor)**2 + (j - anchor)**2) / (2 * sigma2))
            total += kernel[i, j]

    return kernel / total


def smoothTest():
    saltImg = rgb2gray(cv2.imread('SaltPepperNoise/sp4.jpg'))
    gaussImg = rgb2gray(cv2.imread('GaussNoise/noise1.jpg'))
    multiScaleImg = rgb2gray(cv2.imread('GaussMultiScaleSmooth/5.jpg'))

    # 椒盐噪声图像三种平滑测试
    print('椒盐噪声图像三种平滑测试\n')
    # 平均平滑
    salt1 = meanValueSmooth(saltImg)
    # 高斯平滑
    salt2 = gaussianSmooth(saltImg)
    # 中值平滑
    salt3 = medianSmooth(saltImg)
    cv2.imshow('salt-mean', salt1)
    cv2.imshow('salt-gaussian', salt2)
    cv2.imshow('salt-median', salt3)
    cv2.waitKey(0)
    print('\n')

    # 高斯噪声图像三种平滑测试
    print('高斯噪声图像三种平滑测试\n')
    # 平均平滑
    gauss1 = meanValueSmooth(gaussImg)
    # 高斯平滑
    gauss2 = gaussianSmooth(gaussImg)
    # 中值平滑
    gauss3 = medianSmooth(gaussImg)
    cv2.imshow('gauss-mean', gauss1)
    cv2.imshow('gauss-gaussian', gauss2)
    cv2.imshow('gauss-median', gauss3)
    cv2.waitKey(0)
    print('\n')

    # 多尺度高斯平滑测试
    print('多尺度高斯平滑测试\n')
    kernel5 = constructGaussKernel(5)
    kernel9 = constructGaussKernel(9)
    kernel13 = constructGaussKernel(13)
    kernel15 = constructGaussKernel(15)
    # 多尺度平滑图像测试
    mScaleSmooth5 = np.round(
        cv2.filter2D(multiScaleImg, cv2.CV_64F, kernel5, anchor=(2, 2), borderType=cv2.BORDER_ISOLATED)).astype(uint8)
    mScaleSmooth9 = np.round(
        cv2.filter2D(multiScaleImg, cv2.CV_64F, kernel9, anchor=(4, 4), borderType=cv2.BORDER_ISOLATED)).astype(uint8)
    mScaleSmooth13 = np.round(
        cv2.filter2D(multiScaleImg, cv2.CV_64F, kernel13, anchor=(6, 6), borderType=cv2.BORDER_ISOLATED)).astype(uint8)
    mScaleSmooth15 = np.round(
        cv2.filter2D(multiScaleImg, cv2.CV_64F, kernel15, anchor=(7, 7), borderType=cv2.BORDER_ISOLATED)).astype(uint8)
    cv2.imshow('(5, 5)', mScaleSmooth5)
    cv2.imshow('(9, 9)', mScaleSmooth9)
    cv2.imshow('(13, 13)', mScaleSmooth13)
    cv2.imshow('(15, 15)', mScaleSmooth15)
    cv2.waitKey(0)
    # 高斯噪声图像测试
    gaussSmooth5 = np.round(
        cv2.filter2D(gaussImg, cv2.CV_64F, kernel5, anchor=(2, 2), borderType=cv2.BORDER_ISOLATED)).astype(uint8)
    gaussSmooth9 = np.round(
        cv2.filter2D(gaussImg, cv2.CV_64F, kernel9, anchor=(4, 4), borderType=cv2.BORDER_ISOLATED)).astype(uint8)
    gaussSmooth13 = np.round(
        cv2.filter2D(gaussImg, cv2.CV_64F, kernel13, anchor=(6, 6), borderType=cv2.BORDER_ISOLATED)).astype(uint8)
    gaussSmooth15 = np.round(
        cv2.filter2D(gaussImg, cv2.CV_64F, kernel15, anchor=(7, 7), borderType=cv2.BORDER_ISOLATED)).astype(uint8)
    cv2.imshow('(5, 5)', gaussSmooth5)
    cv2.imshow('(9, 9)', gaussSmooth9)
    cv2.imshow('(13, 13)', gaussSmooth13)
    cv2.imshow('(15, 15)', gaussSmooth15)
    cv2.waitKey(0)
    print('\n')

    # 多尺度中值滤波测试
    print('多尺度中值滤波测试\n')
    sp3 = medianSmooth(saltImg, (3, 3))
    sp5 = medianSmooth(saltImg, (5, 5))
    sp7 = medianSmooth(saltImg, (7, 7))
    sp9 = medianSmooth(saltImg, (9, 9))
    cv2.imshow('(3, 3)', sp3)
    cv2.imshow('(5, 5)', sp5)
    cv2.imshow('(7, 7)', sp7)
    cv2.imshow('(9, 9)', sp9)
    cv2.waitKey(0)
    print('\n')


if __name__ == '__main__':
    print('Hello world')
    smoothTest()
