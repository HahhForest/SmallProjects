"""
SJTU-AU333-数字图像处理-作业3形态学
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import os
import imageio

# 二值化图片目录
bipath = './Morphology-Bi'
# 灰度图片目录
graypath = './Morphology-gray'
# 结果输出目录
respath = './MorphRes'
if not os.path.exists(respath):
    os.makedirs(respath)


def bigraph1():
    """
    1-测试腐蚀、膨胀、开、闭运算
    """
    path = os.path.join(bipath, '1.bmp')
    img = cv2.imread(path)

    # 结构元
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # 腐蚀
    erodeRes = cv2.erode(img, kernel, iterations=5)
    # 膨胀
    dilateRes = cv2.dilate(img, kernel, iterations=5)
    # 开运算
    openRes = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)
    # 闭运算
    closeRes = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)

    # cv2.imshow('erodeRes', erodeRes)
    # cv2.imshow('dilateRes', dilateRes)
    # cv2.imshow('openRes', openRes)
    # cv2.imshow('closeRes', closeRes)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(respath, 'bi1-erode.jpg'), erodeRes)
    cv2.imwrite(os.path.join(respath, 'bi1-dilate.jpg'), dilateRes)
    cv2.imwrite(os.path.join(respath, 'bi1-open.jpg'), openRes)
    cv2.imwrite(os.path.join(respath, 'bi1-close.jpg'), closeRes)


def bigraph3():
    """
    3-去噪
    闭运算+开运算，结构元尺寸比噪声大
    """
    path = os.path.join(bipath, '3.bmp')
    img = cv2.imread(path)

    # 结构元
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

    res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv2.imshow('res', res)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(respath, 'bi3-deNoise.jpg'), res)


def bigraph4():
    """
    4-去除杆状结构
    使用圆形结构元开运算
    """
    path = os.path.join(bipath, '4.bmp')
    img = cv2.imread(path)

    # # 结构元，9*9圆形
    # kernel = np.mat([[0,0,0,1,1,1,0,0,0], [0,0,1,1,1,1,1,0,0], [0,1,1,1,1,1,1,1,0],
    #                  [1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1],
    #                  [0,0,0,1,1,1,0,0,0], [0,0,1,1,1,1,1,0,0], [0,1,1,1,1,1,1,1,0]]).astype(np.uint8)

    # 使用开运算
    res = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    # cv2.imshow('circle', res)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(respath, 'bi4-deStock.jpg'), res)


def bigraph6():
    """
    6-分离出水平垂直杆子
    使用对应结构的结构元开运算
    """
    path = os.path.join(bipath, '6.bmp')
    img = cv2.imread(path)

    # 结构元，矩形
    verticalKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
    parallelKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))

    # 使用开运算
    paraRes = cv2.morphologyEx(img, cv2.MORPH_OPEN, parallelKernel, iterations=1)
    vertiRes = cv2.morphologyEx(img, cv2.MORPH_OPEN, verticalKernel, iterations=1)
    # cv2.imshow('paraRes', paraRes)
    # cv2.imshow('vertiRes', vertiRes)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(respath, 'bi6-parallel.jpg'), paraRes)
    cv2.imwrite(os.path.join(respath, 'bi6-vertical.jpg'), vertiRes)


def bigraph8():
    """
    8-从左图得到右侧两个结果
    分别进行顶帽运算、黑帽运算：f - (f open b), (f close b) - f
    """
    path = os.path.join(bipath, '8.jpg')
    img = cv2.imread(path)

    # 结构元
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    # 开运算
    openRes = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    # 闭运算
    closeRes = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 顶帽运算
    tophatRes = cv2.addWeighted(img, 1, openRes, -1, 0)
    # 黑帽运算
    bottomhatRes = cv2.addWeighted(closeRes, 1, img, -1, 0)

    # cv2.imshow('tophatRes', tophatRes)
    # cv2.imshow('bottomhatRes', bottomhatRes)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(respath, 'bi8-left.jpg'), tophatRes)
    cv2.imwrite(os.path.join(respath, 'bi6-right.jpg'), bottomhatRes)


def bigraph9():
    """
    9-修复区域
    使用闭操作
    """
    path = os.path.join(bipath, '9.bmp')
    img = cv2.imread(path)

    # 结构元
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # 闭运算
    closeRes = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # cv2.imshow('closeRes', closeRes)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(respath, 'bi9-fix.jpg'), closeRes)


def gray2():
    """
    2-去除高光区域后分割电话话筒
    """
    path = os.path.join(graypath, '2.gif')
    img = cv2.imread(path)
    # 格式问题，使用imgio读取
    if img is None:
        tmp = imageio.imread(path)
        img = np.array(tmp)

    # 首先二值化
    retval, biRes = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

    # 然后使用闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    openRes = cv2.morphologyEx(biRes, cv2.MORPH_CLOSE, kernel, iterations=3)

    # cv2.imshow('openRes', openRes)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(respath, 'gray2-handset.jpg'), openRes)


def gray3():
    """
    3-形态学滤除黑白噪声点后计算形态学梯度
    使用开运算消除亮杂点，闭运算消除暗杂点
    """
    path = os.path.join(graypath, '3.jpg')
    img = cv2.imread(path)

    # 结构元
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # 开运算
    res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    # 闭运算
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=1)

    # cv2.imshow('res', res)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(respath, 'gray3-deNoise.jpg'), res)


def gray4():
    """
    4-从左图分离得到右侧两图的结果
    分别进行顶帽运算、黑帽运算提取亮、暗处理部：f - (f open b), (f close b) - f
    """
    path = os.path.join(graypath, '4.jpg')
    img = cv2.imread(path)

    # 结构元
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 开运算
    openRes = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    # 闭运算
    closeRes = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 顶帽运算
    tophatRes = cv2.addWeighted(img, 1, openRes, -1, 0)
    # 黑帽运算
    bottomhatRes = cv2.addWeighted(closeRes, 1, img, -1, 0)

    # 二值化
    retval, tophatRes = cv2.threshold(tophatRes, 20, 255, cv2.THRESH_BINARY)
    retval, bottomhatRes = cv2.threshold(bottomhatRes, 60, 255, cv2.THRESH_BINARY)

    # cv2.imshow('tophatRes', tophatRes)
    # cv2.imshow('bottomhatRes', bottomhatRes)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(respath, 'gray4-left.jpg'), tophatRes)
    cv2.imwrite(os.path.join(respath, 'gray3-right.jpg'), bottomhatRes)


if __name__ == '__main__':
    print('Hello world')
    bigraph1()
    bigraph3()
    bigraph4()
    bigraph6()
    bigraph8()
    bigraph9()
    gray2()
    gray3()
    gray4()