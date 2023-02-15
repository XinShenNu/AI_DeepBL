import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
import random

from data_utils import *
from deep_bls import DeepBLS
from MyDataset import *
from active_learning import *
from save_results import *


# 算法主体
def AI_deepBLS(x_train, y_train, init_bins, I, B, once_budget, balancing_AF, bagging_num, sample_rate, batch,
               max_length, num_incre_BLS, **deepBLS_pram):
    starting_time = time.time()
    # 获得初始数据
    batch_size = round(x_train.shape[0] / batch)
    ini_train_size = batch_size
    x_train_init = x_train.iloc[0:ini_train_size]
    y_train_init = y_train.iloc[0:ini_train_size]
    print("初始阶段 : batch_size = %d" % (x_train_init.shape[0]))

    # 初始化直方图，为主动学习做准备
    index = random.sample(range(0, x_train_init.shape[0]), init_bins)
    sampler_features_base = x_train_init.iloc[index].reset_index(drop=True)
    sampler_labels_base = y_train_init.iloc[index].reset_index(drop=True)
    x_train_init = x_train_init.drop(index, axis=0).reset_index(drop=True)
    y_train_init = y_train_init.drop(index, axis=0).reset_index(drop=True)
    hist = MyHist(sampler_labels_base, init_bins)
    # hist.plot_hist()

    # Active learning in initial data
    budget = math.floor(ini_train_size * B) - init_bins
    batch_oracle_annotated_indexs = []
    next_new_train_indexs_list = list(range(0, x_train_init.shape[0]))
    for sess in range(I):
        sess_budget = math.floor(budget * 1.0 / I)
        sess_new_train_indexs = list(
            set(next_new_train_indexs_list) - set(batch_oracle_annotated_indexs))
        if sess == 0:
            sess_sampler_features = sampler_features_base
            sess_sampler_labels = sampler_labels_base
        oracle_annotated_indexs = active_learning(sess, sess_budget, sess_new_train_indexs,
                                                  x_train_init.iloc[sess_new_train_indexs],
                                                  y_train_init.iloc[sess_new_train_indexs],
                                                  balancing_AF, hist,
                                                  sess_sampler_features, sess_sampler_labels,
                                                  once_budget)
        batch_oracle_annotated_indexs.extend(oracle_annotated_indexs)
        print("New Training-set size = " + str(len(oracle_annotated_indexs)))
        print("Training-set size = " + str(len(batch_oracle_annotated_indexs)))
        sess_sampler_features.append(x_train_init.iloc[oracle_annotated_indexs])
        sess_sampler_labels.append(y_train_init.iloc[oracle_annotated_indexs])
        hist.update(y_train_init.iloc[oracle_annotated_indexs])
        # hist.plot_hist()

    x_train_init_sub = x_train_init.iloc[batch_oracle_annotated_indexs].append(sampler_features_base, ignore_index=True)
    y_train_init_sub = y_train_init.iloc[batch_oracle_annotated_indexs].append(sampler_labels_base, ignore_index=True)

    # 在初始数据集上拟合模型
    model = []

    for d in range(bagging_num):
        np.random.seed(d)
        bagging_index = np.random.randint(0, x_train_init_sub.shape[0],
                                          math.ceil(x_train_init_sub.shape[0] * sample_rate)).tolist()
        x_train_init_sub_bagging = x_train_init_sub.iloc[bagging_index].reset_index(drop=True)
        y_train_init_sub_bagging = y_train_init_sub.iloc[bagging_index].reset_index(drop=True)
        base_model = DeepBLS(**deepBLS_pram)
        base_model.fit(np.array(x_train_init_sub_bagging), np.array(y_train_init_sub_bagging))
        model.append(base_model)

    print('初始阶段模型训练完毕！')

    # 更新范例
    sampler_features = x_train_init_sub
    sampler_labels = y_train_init_sub

    if batch > 1:
        # 进入增量阶段
        x_stream = x_train.iloc[ini_train_size:].reset_index(drop=True)
        y_stream = y_train.iloc[ini_train_size:].reset_index(drop=True)

        for i in range(1, batch):
            # 获取增量阶段的数据
            if i == batch - 1:
                x_new_batch = x_stream.iloc[(i - 1) * batch_size:]
                y_new_batch = y_stream.iloc[(i - 1) * batch_size:]
                print("第 %d batch : batch_size = %d" % (i, x_new_batch.shape[0]))
                batch_size = x_new_batch.shape[0]
            else:
                x_new_batch = x_stream.iloc[(i - 1) * batch_size:i * batch_size]
                y_new_batch = y_stream.iloc[(i - 1) * batch_size:i * batch_size]
                print("第 %d batch : batch_size = %d" % (i, x_new_batch.shape[0]))

            # Active learning in incremental data
            budget = math.floor(x_new_batch.shape[0] * B)
            batch_oracle_annotated_indexs = []
            next_new_train_indexs_list = list(range(0, x_new_batch.shape[0]))
            for sess in range(I):
                sess_budget = math.floor(budget * 1.0 / I)
                sess_new_train_indexs = list(
                    set(next_new_train_indexs_list) - set(batch_oracle_annotated_indexs))
                if sess == 0:
                    sess_sampler_features = sampler_features
                    sess_sampler_labels = sampler_labels
                oracle_annotated_indexs = active_learning(sess, sess_budget, sess_new_train_indexs,
                                                          x_new_batch.iloc[sess_new_train_indexs],
                                                          y_new_batch.iloc[sess_new_train_indexs],
                                                          balancing_AF, hist,
                                                          sess_sampler_features, sess_sampler_labels,
                                                          once_budget)
                batch_oracle_annotated_indexs.extend(oracle_annotated_indexs)
                print("New Training-set size = " + str(len(oracle_annotated_indexs)))
                print("Training-set size = " + str(len(batch_oracle_annotated_indexs)))
                sess_sampler_features.append(x_new_batch.iloc[oracle_annotated_indexs])
                sess_sampler_labels.append(y_new_batch.iloc[oracle_annotated_indexs])
                hist.update(y_new_batch.iloc[oracle_annotated_indexs])
                # hist.plot_hist()

            x_new_batch_sub = x_new_batch.iloc[batch_oracle_annotated_indexs]
            y_new_batch_sub = y_new_batch.iloc[batch_oracle_annotated_indexs]

            x_new_and_old = x_new_batch_sub.append(sampler_features, ignore_index=True)
            y_new_and_old = y_new_batch_sub.append(sampler_labels, ignore_index=True)

            # Incremental Update with Fixed Number of BLSs
            for d in range(bagging_num):
                np.random.seed(d)
                bagging_index = np.random.randint(0, x_new_batch_sub.shape[0],
                                                  math.ceil(x_new_batch_sub.shape[0] * sample_rate)).tolist()
                x_new_batch_sub_bagging = x_new_batch_sub.iloc[bagging_index].reset_index(drop=True)
                y_new_batch_sub_bagging = y_new_batch_sub.iloc[bagging_index].reset_index(drop=True)
                base_model = model[d]
                if len(base_model.dBLSs) <= max_length:
                    y_pred_score = np.squeeze(base_model.predict(np.array(x_new_batch_sub_bagging)))
                    base_model.incremental_fit(np.array(x_new_batch_sub_bagging), np.array(y_new_batch_sub_bagging),
                                               y_pred_score, num_incre_BLS)
                    model[d] = base_model
                    print('第{}个基模型已进行增量训练。'.format(d + 1))
                else:
                    print('第{}个基模型已超过最大限制，停止训练！'.format(d + 1))

            if i < batch - 1:
                # 更新范例和直方图
                index = random.sample(range(0, x_new_and_old.shape[0]), init_bins)
                sampler_features_base = x_new_and_old.iloc[index].reset_index(drop=True)
                sampler_labels_base = y_new_and_old.iloc[index].reset_index(drop=True)
                x_candidate_sampler = x_new_and_old.drop(index, axis=0).reset_index(drop=True)
                y_candidate_sampler = y_new_and_old.drop(index, axis=0).reset_index(drop=True)
                hist = MyHist(sampler_labels_base, init_bins)
                # hist.plot_hist()

                budget = math.floor(ini_train_size * B) - init_bins
                batch_oracle_annotated_indexs = []
                next_new_train_indexs_list = list(range(0, x_candidate_sampler.shape[0]))
                for sess in range(I):
                    sess_budget = math.floor(budget * 1.0 / I)
                    sess_new_train_indexs = list(
                        set(next_new_train_indexs_list) - set(batch_oracle_annotated_indexs))
                    if sess == 0:
                        sampler_features = sampler_features_base
                        sampler_labels = sampler_labels_base
                    oracle_annotated_indexs = active_learning(sess, sess_budget, sess_new_train_indexs,
                                                              x_candidate_sampler.iloc[sess_new_train_indexs],
                                                              y_candidate_sampler.iloc[sess_new_train_indexs],
                                                              balancing_AF, hist,
                                                              sampler_features, sampler_labels,
                                                              once_budget)
                    batch_oracle_annotated_indexs.extend(oracle_annotated_indexs)
                    print("New Sampler size = " + str(len(oracle_annotated_indexs)))
                    print("Sampler size = " + str(len(batch_oracle_annotated_indexs)))
                    sampler_features = sampler_features.append(x_candidate_sampler.iloc[oracle_annotated_indexs],
                                                               ignore_index=True)
                    sampler_labels = sampler_labels.append(y_candidate_sampler.iloc[oracle_annotated_indexs],
                                                           ignore_index=True)
                    hist.update(y_candidate_sampler.iloc[oracle_annotated_indexs])
                    # hist.plot_hist()

    training_time = time.time() - starting_time
    # 保存模型
    joblib.dump(model, '../model/model_of_{}.pkl'.format(dataset_name))
    return training_time


# 独立实验
def exp_realworld(dataset_name, num_run, is_differ, time_step, exp_function, init_bins, I, B, once_budget, balancing_AF, bagging_num,
                  sample_rate, **exp_parm):
    aver_total_mae = np.zeros(num_run)
    aver_total_mse = np.zeros(num_run)
    aver_total_rmse = np.zeros(num_run)
    aver_total_mape = np.zeros(num_run)
    aver_total_smape = np.zeros(num_run)
    aver_total_r2 = np.zeros(num_run)
    aver_total_pred = []
    aver_train_time = []
    aver_test_time = []

    for r_seed in range(0, num_run):
        np.random.seed(r_seed)

        # 读取划分数据集
        data = monthly_sunspots()  # 27.8341 3 10 0.1 5 50 -30 bag=10
        # data = ETTh2()  # 1.5020 2 80 0.01 5 80 -30 I=10 once=5 bag=20
        # data = daily_website_visitors()  # 5.3485 2 100 0.01 5 30 -10 I=10 bag=20
        # data = Heart_rate2()  # 1.1390 3 140 0.01 3 50 -30 bag=10
        # data = NASDAQ()  # 0.6915 2 70 0.1 3 10 -10 bag=10
        # data = GOOGL()  # 0.7272 2 80 0.01  3 10 -30 I=10 bag=10
        # data = DowJones()  # 0.3679 3 80 0.01 3 10 -30 bag=10
        # data = Beijing_PM_2_5(time_step)  # 22.1591% 2 10 0.1 20 5 10 -10 step=3 once=20 bag=10
        # data = household_power_consumption(time_step)  # 22.3707 3 40 0.1 20 10 10 -30 step=3 once=3 bag=10
        # data = Gold_Price(time_step)  # 0.4981 3 20 0.1 18 5 30 -10 step=3 bag=15

        if is_differ:
            x_data = data.iloc[:, 2:]
            y_data = data.iloc[:, 0:2]
        else:
            x_data = data.iloc[:, 1:]
            y_data = data.iloc[:, 0]
        x_train, x_test, y_train, y_test = timeseries_train_test_split(x_data, y_data, test_size=0.1)

        if mode == 'validate':
            x_train, x_test, y_train, y_test = timeseries_train_test_split(x_train, y_train, test_size=0.1)
            print('验证模式')
        else:
            print('测试模式')

        if is_differ:
            y_train_differ = pd.DataFrame(y_train['differ'])
            y_train = y_train['y']
            y_test_differ = pd.DataFrame(y_test['differ'])
            y_test = y_test['y']
            y_test = np.array(y_test).reshape(y_test.shape[0], ) + np.array(y_test_differ).reshape(y_test_differ.shape[0], )

        # 模型训练、预测、评价
        tqdm.write('=' * 20)
        tqdm.write((dataset_name + '第' + str(r_seed + 1) + '次运行').center(20))
        training_time = exp_function(x_train, y_train, init_bins, I, B, once_budget, balancing_AF, bagging_num,
                                     sample_rate, **exp_parm)
        model = joblib.load('../model/model_of_{}.pkl'.format(dataset_name))

        # y_pred_score = model.predict(np.array(x_test))
        #
        # num_BLS_before_pruning = len(model.dBLSs)
        # BLS_residual = model.best_BLS_pruning(np.array(y_test))
        # num_BLS_after_pruning = len(model.dBLSs)
        #
        # plt.figure()
        # plt.plot(pd.Series(np.array(BLS_residual)), label='residual')
        # plt.plot(num_BLS_after_pruning - 1, BLS_residual[num_BLS_after_pruning - 1], 'r*')
        # plt.title('residual of each BLS in fine turn of test set')
        # plt.legend(loc='best')
        # plt.tight_layout()

        starting_time = time.time()
        pred_bagging = []
        for d in range(bagging_num):
            y_pred = model[d].predict(np.array(x_test))
            if is_differ:
                y_pred = np.array(y_pred).reshape(y_pred.shape[0], ) + np.array(y_test_differ).reshape(
                    y_test_differ.shape[0], )
            pred_bagging.append(y_pred)
        pred = np.array(pred_bagging)
        pred = np.mean(pred, axis=0)
        testing_time = time.time() - starting_time
        aver_total_pred.append(pred)
        aver_train_time.append(training_time)
        aver_test_time.append(testing_time)
        aver_total_mae[r_seed] = metrics.mean_absolute_error(np.array(y_test).reshape(-1, 1), pred.reshape(-1, 1))
        tqdm.write('Current r_seed mae:' + str(aver_total_mae[r_seed]))
        aver_total_mse[r_seed] = metrics.mean_squared_error(np.array(y_test).reshape(-1, 1), pred.reshape(-1, 1))
        tqdm.write('Current r_seed mse:' + str(aver_total_mse[r_seed]))
        aver_total_rmse[r_seed] = math.sqrt(
            metrics.mean_squared_error(np.array(y_test).reshape(-1, 1), pred.reshape(-1, 1)))
        tqdm.write('Current r_seed rmse:' + str(aver_total_rmse[r_seed]))
        aver_total_mape[r_seed] = mean_absolute_percentage_error(np.array(y_test).reshape(-1, 1), pred.reshape(-1, 1))
        tqdm.write('Current r_seed mape:' + str(aver_total_mape[r_seed]))
        aver_total_smape[r_seed] = symmetric_mean_absolute_percentage_error(np.array(y_test).reshape(-1, 1),
                                                                            pred.reshape(-1, 1))
        tqdm.write('Current r_seed smape:' + str(aver_total_smape[r_seed]))
        aver_total_r2[r_seed] = metrics.r2_score(np.array(y_test).reshape(-1, 1), pred.reshape(-1, 1))
        tqdm.write('Current r_seed r2:' + str(aver_total_r2[r_seed]))

        plt.figure()
        plt.plot(pd.Series(np.array(y_test)), label='actual')
        plt.plot(pd.Series(np.squeeze(pred)), 'g', label='prediction')
        plt.title("Mean absolute percentage error : %.2f%%" % aver_total_mape[r_seed])
        plt.legend(loc='best')
        plt.tight_layout()

    save_to_excel('../output/results_of_{}_AI_deepBLS.xlsx'.format(dataset_name), aver_total_pred, aver_total_mae, aver_total_mse,
                  aver_total_rmse, aver_total_mape, aver_total_smape, aver_total_r2, aver_train_time,
                  aver_test_time)

    tqdm.write('Average mae:' + str(np.mean(aver_total_mae)))
    tqdm.write('Average mse:' + str(np.mean(aver_total_mse)))
    tqdm.write('Average rmse:' + str(np.mean(aver_total_rmse)))
    tqdm.write('Average mape:' + str(np.mean(aver_total_mape)))
    tqdm.write('Average smape:' + str(np.mean(aver_total_smape)))
    tqdm.write('Average r2:' + str(np.mean(aver_total_r2)))
    tqdm.write('Std rmse:' + str(np.std(aver_total_rmse)))
    # plt.show()


# #################################算法参数设置###################################
# 数据集参数
dataset_name = 'monthly_sunspots'

# 实验参数
num_run = 1
mode = 'test'  # test/validate

# 数据预处理参数
is_differ = False

# 数据集参数
time_step = 3

# 弹性策略参数
AI_deepBLS_parm = {
    # 主动学习参数
    'init_bins': 100,
    'I': 5,  # 主动学习迭代次数
    'B': 0.8,  # 主动学习预算
    'once_budget': 1,  # 样本采集函数每次采集样本数量
    'balancing_AF': 'weight',
    # bagging参数
    'bagging_num': 10,  # bagging集成基学习器数目
    'sample_rate': 0.9,  # bagging采样率
    # 增量学习参数
    'batch': 3,  # 增量阶段数量（含初始阶段）
    'max_length': 10000,  # 模型最大尺寸
    'num_incre_BLS': 10  # 每个增量阶段拟合的新BLS数量
}

# DeepBL参数
deepBLS_pram = {
    'max_iter': 10,  # 模型初始阶段拟合的BLS数量  # 21.5347
    'learn_rate': 0.1,
    'NumFea': 20,
    'NumWin': 5,
    'NumEnhan': 50,
    's': 0.8,
    'C': 2 ** -30
}

# ###################################算法执行####################################
starting_time = time.time()
AI_deepBLS_parm.update(deepBLS_pram)
exp_realworld(dataset_name, num_run, is_differ, time_step, AI_deepBLS, **AI_deepBLS_parm)
running_time = time.time() - starting_time
print(timedelta(seconds=round(running_time)))
