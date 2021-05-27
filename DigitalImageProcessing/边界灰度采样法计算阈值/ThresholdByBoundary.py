"""
使用边界灰度值聚类的方法找到最佳阈值
参考论文 "Lisheng Wang, Jing Bai, Threshold selection by clustering gray levels of boundary,
Pattern Recognition Letters 24 (2003) 1983–1999"
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import *

# 支持中文标题
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置


class ImgProcessor:
    """
    图像处理器
    在opencv中，图像以numpy.mat类型储存，因此打印尺寸信息为先高后宽。但在此类中，所有涉及到尺寸的表达方式均为先宽后高
    """
    def __init__(self, img, doGaussian=False, gaussKSize=(3, 3), gradientOpe='sobel', gTh=0):
        """
        :param img: 读入的图像数据，类型为numpy.mat，三通道彩色图像
        """
        print('初始化...')

        # 输入合法性检测
        operatorLst = ['robert', 'canny', 'prewitt', 'sobel']
        if gradientOpe not in operatorLst:
            raise ValueError('算子错误，尝试使用\'robert\', \'canny\', \'prewitt\', \'sobel\'')

        # 超参数
        # 是否进行高斯模糊
        self.doGaussian = doGaussian
        # 高斯模糊核尺寸，必须是正奇数
        self.gaussKSize = gaussKSize
        # 梯度计算的算子种类
        self.gradientOpe = gradientOpe
        # 找边界时的梯度阈值，设为0则为不使用梯度阈值约束
        self.gTh = gTh

        # 图像信息
        self.rgb = img
        # 图像尺寸
        self.width = img.shape[1]
        self.height = img.shape[0]
        # 填充后图像尺寸
        self.widthFilled = self.width + 2
        self.heightFilled = self.height + 2
        self.gray = self.rgb2gray()
        if self.doGaussian:
            self.gray = self.gaussianFilter()
        self.grayFilled = self.__edgeFill()

        # 信息打印
        initLog = '初始化成功，超参数：高斯模糊-'
        if doGaussian:
            initLog = initLog + 'kSize-' +str(self.gaussKSize)
        else:
            initLog = initLog + '无'
        initLog = initLog + ', 算子-' + self.gradientOpe + ', 梯度阈值-' + str(self.gTh)
        print(initLog)

    def __edgeFill(self):
        """
        对灰度图像在四周填充一圈空像素进行扩充
        :return: 返回扩充后的图像，类型为numpy.mat
        """
        res = mat(np.zeros((self.heightFilled, self.widthFilled), dtype=uint8))
        for i in range(1, self.height + 1):
            for j in range(1, self.width + 1):
                res[i, j] = self.gray[i-1, j-1]
        for j in range(0, self.widthFilled):
            res[0, j] = 0
            res[self.height + 1, j] = 0
        for i in range(1, self.height + 1):
            res[i, 0] = 0
            res[i, self.width + 1] = 0
        return res

    def __gradient2(self):
        """
        使用尺寸为2的算子计算图像的梯度矩阵，对齐方式为左上角
        由于算子尺寸为偶数，因此依照定义，卷积核锚点设置为(0, 0)
        梯度幅度矩阵目的不是用来显示，因此若要显示查看效果，需将datatype改为uint8
        :param ope: 使用的算子
        :return: 图像的梯度幅度矩阵与方向矩阵，与原图像尺寸相同
        """
        kernelX = []
        kernelY = []
        theta = mat(np.zeros((self.height, self.width)))

        if self.gradientOpe == 'robert':
            kernelX = mat(np.array([[1, 0], [0, -1]]))
            kernelY = mat(np.array([[0, -1], [1, 0]]))
        elif self.gradientOpe == 'canny':
            kernelX = mat(np.array([[-1, 1], [1, 1]]))
            kernelY = mat(np.array([[-1, 1], [-1, -1]]))

        # 存浮点型，保证计算梯度方向时的精确度。使用cv2自带黑色补全
        gradientX = cv2.filter2D(self.gray, cv2.CV_16S, kernelX, anchor=(0, 0), borderType=cv2.BORDER_ISOLATED).astype(np.float16)
        gradientY = cv2.filter2D(self.gray, cv2.CV_16S, kernelY, anchor=(0, 0), borderType=cv2.BORDER_ISOLATED).astype(np.float16)
        for i in range(self.height):
            for j in range(self.width):
                theta[i, j] = math.atan2(gradientX[i, j], gradientY[i, j])

        # 以绝对值之和近似求梯度值
        gradientX = mat(np.maximum(gradientX, -gradientX))
        gradientY = mat(np.maximum(gradientY, -gradientY))
        magnitude = gradientX * 0.5 + gradientY * 0.5

        return magnitude, theta

    def __gradient3(self):
        """
        使用尺寸为3的算子计算图像的梯度矩阵，对齐方式为算子中心，即卷积核锚点设置为(1, 1)
        梯度幅度矩阵目的不是用来显示，因此若要显示查看效果，需将datatype改为uint8
        :param ope: 使用的算子
        :return: 图像的梯度幅度矩阵与方向矩阵，与原图像尺寸相同
        """
        kernelX = []
        kernelY = []
        theta = mat(np.zeros((self.height, self.width)))

        if self.gradientOpe == 'prewitt':
            kernelX = mat(np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
            kernelY = mat(np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
        elif self.gradientOpe == 'sobel':
            kernelX = mat(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
            kernelY = mat(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))

        # 存浮点型，保证计算梯度方向时的精确度。使用cv2自带黑色补全
        gradientX = cv2.filter2D(self.gray, cv2.CV_16S, kernelX, anchor=(1, 1), borderType=cv2.BORDER_ISOLATED).astype(np.float16)
        gradientY = cv2.filter2D(self.gray, cv2.CV_16S, kernelY, anchor=(1, 1), borderType=cv2.BORDER_ISOLATED).astype(np.float16)
        for i in range(self.height):
            for j in range(self.width):
                theta[i, j] = math.atan2(gradientX[i, j], gradientY[i, j])

        # 以绝对值之和近似求梯度值
        gradientX = mat(np.maximum(gradientX, -gradientX))
        gradientY = mat(np.maximum(gradientY, -gradientY))
        magnitude = gradientX * 0.5 + gradientY * 0.5

        # cv2.imshow('mag', magnitude.astype(uint8))
        # cv2.waitKey(100000)
        return magnitude, theta

    def changeImg(self, img):
        """
        更换图片，重新设置图像处理器的参数
        :param img: 新图像
        """
        print('更换图片...')
        self.rgb = img
        # 图像尺寸
        self.width = img.shape[1]
        self.height = img.shape[0]
        # 填充后图像尺寸
        self.widthFilled = self.width + 2
        self.heightFilled = self.height + 2
        self.gray = self.rgb2gray()
        if self.doGaussian:
            self.gray = self.gaussianFilter()
        self.grayFilled = self.__edgeFill()

    def setHyperPara(self, doGaussian=False, gaussKSize=(3, 3), gradientOpe='sobel', gTh=0):
        """
        设置超参数，可以在一个实例中重复设置。
        :param doGaussian: 是否进行高斯滤波
        :param gaussKSize: 高斯滤波的核尺寸，必须是正奇数元组
        :param gradientOpe: 计算梯度阈值的算子，默认sobel
        :param gTh: 梯度约束的阈值，默认为0，即不使用梯度阈值约束
        """
        print('更改超参数...')
        # 输入合法性检测
        operatorLst = ['robert', 'canny', 'prewitt', 'sobel']
        if gradientOpe not in operatorLst:
            raise ValueError('算子错误，尝试使用\'robert\', \'canny\', \'prewitt\', \'sobel\'')
        # 是否进行高斯模糊
        self.doGaussian = doGaussian
        # 高斯模糊核尺寸，必须是正奇数
        self.gaussKSize = gaussKSize
        # 梯度计算的算子种类
        self.gradientOpe = gradientOpe
        # 找边界时的梯度阈值，设为0则为不使用梯度阈值约束
        self.gTh = gTh

        # 修改灰度图
        self.gray = self.rgb2gray()
        if doGaussian:
            self.gray = self.gaussianFilter()

        # 信息打印
        initLog = '更改超参数：高斯模糊-'
        if doGaussian:
            initLog = initLog + 'kSize-' + str(self.gaussKSize)
        else:
            initLog = initLog + '无'
        initLog = initLog + ', 算子-' + self.gradientOpe + ', 梯度阈值-' + str(self.gTh)
        print(initLog)

    def rgb2gray(self):
        """
        彩色图像转为灰度图像，使用人眼的敏感度
        :return: 灰度图像，类型为num.mat
        """
        print('灰度化...')
        gray = mat(zeros((self.height, self.width), dtype=uint8))
        # 已经是灰度图
        if size(self.rgb.shape) == 2:
            return self.rgb
        # 单通道灰度图
        if self.rgb.shape[2] == 1:
            for i in range(self.height):
                for j in range(self.width):
                    gray[i, j] = self.rgb[i, j, 0]

        for i in range(self.height):
            for j in range(self.width):
                gray[i, j] = 0.11*self.rgb[i, j, 0] + 0.59*self.rgb[i, j, 1] + 0.3*self.rgb[i, j, 2]
        return gray

    def gaussianFilter(self):
        """
        高斯滤波，调用cv2.GaussianBlur()实现。源为
        :param kSize: 高斯核尺寸，width和height必须为正奇数
        :return: 滤波后的图像，和原函数同尺寸
        """
        print('高斯滤波...')
        res = mat(zeros((self.height, self.width), dtype=uint8))
        cv2.GaussianBlur(self.gray, self.gaussKSize, sigmaX=0, dst=res)
        return res

    def laplasian(self):
        """
        使用拉普拉斯算子计算图像的二阶导，使用填充过的图像
        计算出的拉普拉斯矩阵目的不是为了显示。若要使用imshow()显示应先将元素类型转换为uint8
        :return: 拉普拉斯矩阵，尺寸与原图像相同
        """
        print('拉普拉斯滤波...')
        # 拉普拉斯卷积模板
        lapFlt = mat(np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
        res = cv2.filter2D(self.gray, cv2.CV_16S, lapFlt, anchor=(1, 1), borderType=cv2.BORDER_ISOLATED)

        return res

    def gradient(self):
        """
        计算图像的梯度，可以使用不同的算子，默认sobel
        :return: 梯度矩阵，与原图像同尺寸
        """
        print('梯度矩阵求解：算子-' + self.gradientOpe)
        if self.gradientOpe == 'robert' or self.gradientOpe == 'canny':
            return self.__gradient2()
        elif self.gradientOpe == 'prewitt' or self.gradientOpe == 'sobel':
            return self.__gradient3()

    def grayValueAlongBoundary(self):
        """
        使用梯度约束的拉普拉斯检测得到边界
        考虑相邻的四个像素组成的小方格。对其四条边进行检测是否与边界相交。若出现二阶导即拉普拉斯值零交叉且梯度之和大于等于梯度阈值，认为与边界相交
        若一个小方格中有至少两条边检测相交，则认为此方格为“边界方格”。通过插值计算相交点的灰度值。绘制相交点的灰度值直方图，即可看到对应的最佳阈值
        设置梯度阈值gTh <= 0即为不使用梯度阈值
        :return: 灰度值列表，可以用来取平均求最优阈值和画直方图
        """
        # 边界灰度值采样列表
        print('边界灰度值采样：梯度阈值-' + str(self.gTh))
        sampleLst = []
        # 拉普拉斯矩阵、梯度值矩阵、梯度方向矩阵
        lapMat = self.laplasian()
        gMagMat, gDirMat = self.gradient()

        # 横向边检测，遍历左端点
        for i in range(self.height):
            for j in range(self.width - 1):
                if ((lapMat[i, j] > 0 and lapMat[i, j + 1] < 0) or (lapMat[i, j] < 0 and lapMat[i, j + 1] > 0))\
                        and gMagMat[i, j] + gMagMat[i, j + 1] >= self.gTh:
                    w = float(abs(lapMat[i, j + 1])) / (abs(lapMat[i, j]) + abs(lapMat[i, j + 1]))
                    sampleLst.append(self.gray[i, j] * w + self.gray[i, j + 1] * (1-w))

        # 纵向边检测，遍历上端点
        for j in range(self.width):
            for i in range(self.height - 1):
                if ((lapMat[i, j] > 0 and lapMat[i + 1, j] < 0) or (lapMat[i, j] < 0 and lapMat[i + 1, j] > 0)) \
                        and gMagMat[i, j] + gMagMat[i + 1, j] >= self.gTh:
                    w = float(abs(lapMat[i + 1, j])) / (abs(lapMat[i, j]) + abs(lapMat[i + 1, j]))
                    sampleLst.append(self.gray[i, j] * w + self.gray[i + 1, j] * (1 - w))

        print('采样点个数-' + str(size(sampleLst)) + ', 均值-' + str(mean(sampleLst)))
        return sampleLst

    def drawHistogram(self, normed=True):
        """
        绘制直方图，分别为图像的灰度直方图以及边界的灰度直方图
        :param normed: 纵坐标是否使用密度比例
        """
        print('绘制直方图...')
        boundaryGrayLst = self.grayValueAlongBoundary()
        grayRange = list(range(256))

        plt.figure('梯度算子-' + self.gradientOpe + ', 梯度阈值-' + str(self.gTh))
        # 图像的灰度直方图子图
        plt.subplot(2, 1, 1)
        imgGrayLst = list(self.gray.flatten().A)
        plt.hist(imgGrayLst, bins=grayRange, density=normed, color='blue')
        plt.title('图像的灰度直方图')

        # 边界灰度直方图子图
        plt.subplot(2, 1, 2)
        plt.hist(boundaryGrayLst, bins=grayRange, density=normed, color='green')
        plt.title('边界的灰度直方图')

        # 自动调整子图距离
        plt.tight_layout()
        plt.show()

    def binaryzation(self, threshold):
        """
        使用给定的阈值进行二值化
        :param threshold: 阈值
        :return: 二值化图像，与原图像同尺寸
        """
        biGph = self.gray.copy()
        biGph[biGph >= threshold] = 255
        biGph[biGph < threshold] = 0
        return biGph


def unitTest():
    """
    单元测试，测试各个功能
    """
    processor = ImgProcessor(cv2.imread('1_gray.bmp'), doGaussian=True, gaussKSize=(7, 7), gradientOpe='sobel', gTh=200)
    print(processor.rgb.shape)
    # 灰度化、尺寸填充测试
    cv2.imshow('gray', processor.gray)

    # 拉普拉斯测试
    lap = processor.laplasian()
    lap[lap < 0] = 0
    lap[lap > 255] = 255
    lap = lap.astype(uint8)
    print(lap.dtype)
    print(lap)
    cv2.imshow('lap', lap)
    kernel = mat(np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
    lapCV2 = cv2.filter2D(processor.gray, -1, kernel, anchor=(1, 1))
    cv2.imshow('lapCV2', lapCV2)
    cv2.waitKey(0)

    # 梯度计算测试
    print(processor.gradient())

    # 沿边界灰度值采样测试
    grayLst = processor.grayValueAlongBoundary()
    print(len(grayLst), np.mean(grayLst))
    # 绘制直方图测试
    processor.drawHistogram()

def hwTest():
    """
    按照作业要求测试
    """
    print('实验开始\n')
    processor = ImgProcessor(cv2.imread('22.bmp'), doGaussian=False, gaussKSize=(3, 3), gradientOpe='sobel', gTh=0)
    print('\n')

    print('阈值分割算法结果复现\n')

    print('\n')

    print('不同梯度阈值对比\n')
    for gTh in [100, 150, 200, 250, 300, 350]:
        processor.setHyperPara(gTh=gTh)
        processor.drawHistogram(normed=True)
    print('\n')

    print('不同梯度计算模型对比\n')
    gradientOpeLst = ['robert', 'canny', 'prewitt', 'sobel']
    optGThLst = [50, 420, 200, 300]
    for k in range(4):
        processor.setHyperPara(gradientOpe=gradientOpeLst[k], gTh=optGThLst[k])
        processor.drawHistogram(normed=True)
    print('\n')

    print('梯度约束的Laplacian检测与直接Laplacian检测对比\n')
    processor.setHyperPara(gradientOpe='sobel', gTh=300)
    processor.drawHistogram(normed=True)
    processor.setHyperPara(gradientOpe='sobel', gTh=0)
    processor.drawHistogram(normed=True)
    print('\n')

    print('进行高斯滤波降噪与不进行对比\n')
    processor.setHyperPara(doGaussian=False, gradientOpe='sobel', gTh=300)
    processor.drawHistogram(normed=True)
    processor.setHyperPara(doGaussian=True, gaussKSize=(3, 3), gradientOpe='sobel', gTh=300)
    processor.drawHistogram(normed=True)
    processor.setHyperPara(doGaussian=True, gaussKSize=(5, 5), gradientOpe='sobel', gTh=200)
    processor.drawHistogram(normed=True)
    processor.setHyperPara(doGaussian=True, gaussKSize=(7, 7), gradientOpe='sobel', gTh=200)
    processor.drawHistogram(normed=True)
    print('\n')


if __name__ == '__main__':
    print("Hello world!")
    hwTest()