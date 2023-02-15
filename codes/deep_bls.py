import numpy as np
import matplotlib.pyplot as plt

from BLS_Regression import *


class DeepBLS(object):
    # 模型初始化
    def __init__(self,
                 max_iter=50,
                 learn_rate=0.01,
                 new_BLS_max_iter=10,
                 NumFea=20,
                 NumWin=5,
                 NumEnhan=50,
                 s=0.8,
                 C=2 ** -30):

        self.max_iter = max_iter  # 模型中当前BLS的数量
        self.learn_rate = learn_rate
        self.new_BLS_max_iter = new_BLS_max_iter  # 每个增量阶段拟合的新BLS数量 num_incre_BLS
        self.NumFea = NumFea
        self.NumWin = NumWin
        self.NumEnhan = NumEnhan
        self.s = s
        self.C = C
        self.dBLSs = []
        self.residual_mean = None  # 每个BLS的残差均值
        self.cumulated_pred_score = None  # 模型剪枝到每个BLS对数据集的最终预测值

    # 模型初始阶段拟合
    def fit(self, x_train, y_train):

        n, m = x_train.shape  # 训练集行、列
        self.residual_mean = np.zeros(self.max_iter)
        loss = np.zeros(self.max_iter)

        # 训练第一个BLS，计算初始f值
        dBLS = BLS(self.NumFea, self.NumWin, self.NumEnhan)
        dBLS.train(x_train, np.array(y_train).reshape(-1, 1), self.s, self.C)
        self.dBLSs.append(dBLS)
        f = np.array(dBLS.test(x_train)).reshape(x_train.shape[0],)
        loss[0] = mean_absolute_percentage_error(y_train, f)

        # 拟合初始阶段每个BLS
        for iter_ in range(1, self.max_iter):
            y_predict = f

            residual = y_train - y_predict  # 计算每个BLS的残差

            dBLS = BLS(self.NumFea, self.NumWin, self.NumEnhan)
            # fit to negative gradient
            dBLS.train(x_train, np.array(residual * self.learn_rate).reshape(-1, 1), self.s,
                        self.C)  # 用当前采样集和学习率乘以残差拟合一个BLS
            self.dBLSs.append(dBLS)  # append new BLS

            f += np.array(dBLS.test(x_train)).reshape(x_train.shape[0],)

            loss[iter_] = mean_absolute_percentage_error(y_train, f)

        plt.figure()
        plt.plot(pd.Series(np.array(loss)), label='loss')
        plt.title("loss of each BLS")
        plt.legend(loc='best')
        plt.tight_layout()

    # 使用模型进行预测
    def predict(self, x):

        n = x.shape[0]
        y = np.zeros([n, len(self.dBLSs)])  # 每个BLS的预测值

        for iter_ in range(len(self.dBLSs)):
            dBLS = self.dBLSs[iter_]
            y[:, iter_] = dBLS.test(x).flatten()

        self.cumulated_pred_score = np.cumsum(y, axis=1)  # 按列累加每个BLS的预测值，就是模型剪枝到当前位置的最终的预测值
        return np.sum(y, axis=1)  # 返回整个模型的最终预测值

    # 增量学习
    def incremental_fit(self, x_test, y_test, pred_score, new_BLS_max_iter):

        n, m = x_test.shape
        f = pred_score
        loss = np.zeros(new_BLS_max_iter)

        # 拟合增量阶段的新BLS
        for iter_ in range(new_BLS_max_iter):
            y_residual = y_test - f

            # 使用新数据集和新数据集上的残差拟合新BLS
            new_BLS = BLS(self.NumFea, self.NumWin, self.NumEnhan)
            new_BLS.train(x_test, np.array(y_residual * self.learn_rate).reshape(-1, 1), self.s,
                           self.C)
            self.dBLSs.append(new_BLS)
            self.max_iter += 1

            f += np.array(new_BLS.test(x_test)).reshape(x_test.shape[0],)

            loss[iter_] = mean_absolute_percentage_error(y_test, f)

        plt.figure()
        plt.plot(pd.Series(np.array(loss)), label='loss')
        plt.title("loss of each BLS")
        plt.legend(loc='best')
        plt.tight_layout()
