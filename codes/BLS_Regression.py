import numpy as np
from sklearn import preprocessing
from numpy import random
import time

from data_utils import *


def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def pinv(A, reg):
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = np.dot(A.T, A)
    m = A.shape[1]
    n = b.shape[1]
    wk = np.zeros([m, n], dtype='double')
    ok = np.zeros([m, n], dtype='double')
    uk = np.zeros([m, n], dtype='double')
    L1 = np.mat(AA + np.eye(m)).I
    L2 = np.dot(np.dot(L1, A.T), b)
    for i in range(itrs):
        tempc = ok - uk
        ck = L2 + np.dot(L1, tempc)
        ok = shrinkage(ck + uk, lam)
        uk += ck - ok
        wk = ok
    return wk


class BLS:
    def __init__(self, NumFea, NumWin, NumEnhan):
        self.NumFea = NumFea
        self.NumWin = NumWin
        self.NumEnhan = NumEnhan
        self.WFSparse = list()
        self.distOfMaxAndMin = np.zeros(self.NumWin)
        self.meanOfEachWindow = np.zeros(self.NumWin)
        self.WeightEnhan = None
        self.WeightTop = None

    def train(self, train_x, train_y, s, C):
        # 初始化权值
        u = 0
        WF = list()
        for i in range(self.NumWin):
            random.seed(i + u)
            WeightFea = 2 * random.randn(train_x.shape[1] + 1, self.NumFea) - 1  # 随机初始化从输入到特征映射结点的权值矩阵、偏置矩阵
            WF.append(WeightFea)
        #    random.seed(100)
        self.WeightEnhan = 2 * random.randn(self.NumWin * self.NumFea + 1, self.NumEnhan) - 1  # 随机初始化特征映射结点到增强结点的权值矩阵、偏置矩阵

        # 模型训练
        time_start = time.time()
        H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])])
        y = np.zeros([train_x.shape[0], self.NumWin * self.NumFea])
        # self.WFSparse = list()
        # distOfMaxAndMin = np.zeros(self.NumWin)
        # meanOfEachWindow = np.zeros(self.NumWin)

        for i in range(self.NumWin):
            WeightFea = WF[i]
            A1 = H1.dot(WeightFea)
            scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)  # 数据最大最小归一化
            A1 = scaler1.transform(A1)
            WeightFeaSparse = sparse_bls(A1, H1).T  # 对初始化权重进行微调，获得输入数据的稀疏表示
            self.WFSparse.append(WeightFeaSparse)

            T1 = H1.dot(WeightFeaSparse)  # 计算各特征映射结点的值并归一化
            self.meanOfEachWindow[i] = T1.mean()
            self.distOfMaxAndMin[i] = T1.max() - T1.min()
            T1 = (T1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]
            y[:, self.NumFea * i:self.NumFea * (i + 1)] = T1

        H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])])
        T2 = H2.dot(self.WeightEnhan)  # 计算增强结点的输出
        T2 = tansig(T2)
        T3 = np.hstack([y, T2])
        self.WeightTop = pinv(T3, C).dot(train_y)  # 计算各节点与输出结点之间的连接权重矩阵

        Training_time = time.time() - time_start
        # print('Training has been finished!')
        # print('The Total Training Time is : ', round(Training_time, 6), ' seconds')

        NetoutTrain = T3.dot(self.WeightTop)  # 计算训练后模型的输出值

        RMSE = np.sqrt((NetoutTrain - train_y).T * (NetoutTrain - train_y) / train_y.shape[0])
        # MAPE = sum(abs(NetoutTrain - train_y)) / train_y.mean() / train_y.shape[0]
        MAPE = mean_absolute_percentage_error(np.array(train_y), np.array(NetoutTrain))
        train_ERR = RMSE
        train_MAPE = MAPE
        # print('Training RMSE is : ', RMSE)
        # print('Training MAPE is : ', MAPE)

        return NetoutTrain, train_ERR, train_MAPE, Training_time

    # 模型测试
    def test(self, test_x):
        time_start = time.time()
        HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])
        yy1 = np.zeros([test_x.shape[0], self.NumWin * self.NumFea])
        for i in range(self.NumWin):
            WeightFeaSparse = self.WFSparse[i]
            TT1 = HH1.dot(WeightFeaSparse)
            TT1 = (TT1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]
            yy1[:, self.NumFea * i:self.NumFea * (i + 1)] = TT1

        HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
        TT2 = tansig(HH2.dot(self.WeightEnhan))
        TT3 = np.hstack([yy1, TT2])
        NetoutTest = TT3.dot(self.WeightTop)
        Testing_time = time.time() - time_start

        return NetoutTest
