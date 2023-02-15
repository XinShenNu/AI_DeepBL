import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import LabelEncoder

from myhist import *

register_matplotlib_converters()


def timeseries_train_test_split(X, y, test_size):
    test_index = int(len(X) * (1 - test_size))
    x_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    x_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    return x_train, x_test, y_train, y_test


def series_to_supervised(data, time_step):
    fea_num = data.shape[1]
    columns = list()
    names = list()

    for t in range(1, time_step + 1):
        columns.append(data.shift(t))
        names += [('fea{}_(t-{})'.format(i, t)) for i in range(1, fea_num + 1)]

    supervised = pd.concat(columns, axis=1)
    supervised.columns = names

    return supervised


def monthly_sunspots():
    file_path = '../data/monthly-sunspots.xlsx'
    print("当前数据集：%s" % file_path)
    data = pd.read_excel(file_path)

    # sns.set_style('darkgrid')
    data = pd.DataFrame(data['Sunspots'])
    # plt.figure(figsize=(15, 7))
    # plt.title('monthly-sunspots')
    # plt.plot(data)

    data.columns = ['y']
    for i in range(1, 21):
        data['lag_{}'.format(i)] = data.y.shift(i)

    data = data.dropna()

    hist = MyHist(data['y'], 100)
    # hist.plot_hist()

    return data.reset_index(drop=True)


def ETTh2():
    file_path = '../data/ETTh2.xlsx'
    print("当前数据集：%s" % file_path)
    data = pd.read_excel(file_path)

    sns.set_style('darkgrid')
    data = pd.DataFrame(data['OT'])
    plt.figure(figsize=(15, 7))
    plt.title('ETTh2')
    plt.plot(data)

    data.columns = ['y']
    for i in range(1, 21):
        data['lag_{}'.format(i)] = data.y.shift(i)

    data = data.dropna()

    hist = MyHist(data['y'], 100)
    # hist.plot_hist()

    return data.reset_index(drop=True)


def daily_website_visitors():
    file_path = '../data/daily-website-visitors.csv'
    print("当前数据集：%s" % file_path)
    data = pd.read_csv(file_path)

    # sns.set_style('darkgrid')
    data = pd.DataFrame(data['Unique.Visits'])
    # plt.figure(figsize=(15, 7))
    # plt.title('daily-website-visitors')
    # plt.plot(data)

    data.columns = ['y']
    for i in range(1, 21):
        data['lag_{}'.format(i)] = data.y.shift(i)

    data = data.dropna()

    hist = MyHist(data['y'], 100)
    # hist.plot_hist()

    return data.reset_index(drop=True)


def Heart_rate2():
    file_path = '../data/Heart rate2.csv'
    print("当前数据集：%s" % file_path)
    data = pd.read_csv(file_path, header=None)

    # sns.set_style('darkgrid')
    # plt.figure(figsize=(15, 7))
    # plt.title('Heart rate2')
    # plt.plot(data)

    data.columns = ['y']
    for i in range(1, 21):
        data['lag_{}'.format(i)] = data.y.shift(i)

    data = data.dropna()

    hist = MyHist(data['y'], 100)
    # hist.plot_hist()

    return data.reset_index(drop=True)


def NASDAQ():
    file_path = '../data/NASDAQ.csv'
    print("当前数据集：%s" % file_path)
    data = pd.read_csv(file_path)

    # sns.set_style('darkgrid')
    # plt.figure()
    # plt.title('NASDAQ')
    # plt.plot(data)

    data_tag1 = pd.DataFrame(data.iloc[0:1]).append(data.iloc[0:-1]).reset_index(drop=True)
    data_differ = pd.DataFrame(np.array(data) - np.array(data_tag1))
    # plt.figure(figsize=(30, 5))
    # plt.title('NASDAQ_differ')
    # plt.plot(data_differ, linewidth=0.5)

    data_differ.columns = ['y']
    data.columns = ['y']

    data_differ['differ'] = data_tag1

    for i in range(1, 21):
        data_differ['lag_{}'.format(i)] = data_differ.y.shift(i)

    data_differ = data_differ.dropna()

    hist = MyHist(data['y'], 100)
    # hist.plot_hist()

    hist = MyHist(data_differ['y'], 100)
    hist.plot_hist()

    return data_differ.reset_index(drop=True)


def GOOGL():
    file_path = '../data/GOOGL_2006-01-01_to_2018-01-01.csv'
    print("当前数据集：%s" % file_path)
    data = pd.read_csv(file_path)

    # sns.set_style('darkgrid')
    data = pd.DataFrame(data['Close'])
    # plt.figure()
    # plt.title('GOOGL')
    # plt.plot(data)

    data_tag1 = pd.DataFrame(data.iloc[0:1]).append(data.iloc[0:-1]).reset_index(drop=True)
    data_differ = pd.DataFrame(np.array(data) - np.array(data_tag1))
    # plt.figure(figsize=(30, 5))
    # plt.title('GOOGL_differ')
    # plt.plot(data_differ, linewidth=0.5)

    data_differ.columns = ['y']
    data.columns = ['y']

    data_differ['differ'] = data_tag1

    for i in range(1, 21):
        data_differ['lag_{}'.format(i)] = data_differ.y.shift(i)

    data_differ = data_differ.dropna()

    hist = MyHist(data['y'], 100)
    # hist.plot_hist()

    hist = MyHist(data_differ['y'], 100)
    # hist.plot_hist()

    return data_differ.reset_index(drop=True)


def DowJones():
    file_path = '../data/DowJones.csv'
    print("当前数据集：%s" % file_path)
    data = pd.read_csv(file_path)

    # sns.set_style('darkgrid')
    # plt.figure()
    # plt.title('DowJones')
    # plt.plot(data)

    data_tag1 = pd.DataFrame(data.iloc[0:1]).append(data.iloc[0:-1]).reset_index(drop=True)
    data_differ = pd.DataFrame(np.array(data) - np.array(data_tag1))
    # plt.figure(figsize=(30, 5))
    # plt.title('DowJones_differ')
    # plt.plot(data_differ, linewidth=0.5)

    data_differ.columns = ['y']
    data.columns = ['y']

    data_differ['differ'] = data_tag1

    for i in range(1, 21):
        data_differ['lag_{}'.format(i)] = data_differ.y.shift(i)

    data_differ = data_differ.dropna()

    hist = MyHist(data['y'], 100)
    # hist.plot_hist()

    hist = MyHist(data_differ['y'], 100)
    # hist.plot_hist()

    return data_differ.reset_index(drop=True)


def Beijing_PM_2_5(time_step):
    file_path = '../data/Beijing PM2.5.csv'
    print("当前数据集：%s" % file_path)
    data = pd.read_csv(file_path)

    data.drop(columns=['No', 'year', 'month', 'day', 'hour'], inplace=True)
    data = data.iloc[24:, :]
    data = data.fillna(method='ffill')

    encoder = LabelEncoder()
    data['cbwd'] = encoder.fit_transform(data['cbwd'])

    # sns.set_style('darkgrid')
    target = pd.DataFrame(data['pm2.5'])
    # plt.figure(figsize=(15, 7))
    # plt.title('Beijing PM2.5')
    # plt.plot(target)

    target.columns = ['y']
    for i in range(1, time_step + 1):
        target['lag_{}'.format(i)] = target.y.shift(i)

    data.drop(columns=['pm2.5'], inplace=True)

    feature = series_to_supervised(data, time_step)

    data = pd.concat([target, feature], axis=1)

    data = data.dropna()

    hist = MyHist(data['y'], 100)
    # hist.plot_hist()

    return data.reset_index(drop=True)


def household_power_consumption(time_step):
    file_path = '../data/household_power_consumption.csv'
    print("当前数据集：%s" % file_path)
    data = pd.read_csv(file_path, sep=';', parse_dates={'DT': ['Date', 'Time']}, infer_datetime_format=True,
                       low_memory=False, na_values=['nan', '?'], index_col='DT')
    # data = data.apply(pd.to_numeric, errors='ignore')

    data = data.resample('D').mean()

    # sns.set_style('darkgrid')
    target = pd.DataFrame(data['Global_active_power'])
    # plt.figure(figsize=(15, 7))
    # plt.title('household power consumption')
    # plt.plot(target)

    target.columns = ['y']
    for i in range(1, time_step + 1):
        target['lag_{}'.format(i)] = target.y.shift(i)

    data.drop(columns=['Global_active_power'], inplace=True)

    feature = series_to_supervised(data, time_step)

    data = pd.concat([target, feature], axis=1)

    data = data.dropna()

    hist = MyHist(data['y'], 100)
    hist.plot_hist()

    return data.reset_index(drop=True)


def Gold_Price(time_step):
    file_path = '../data/Gold Price.csv'
    print("当前数据集：%s" % file_path)
    data = pd.read_csv(file_path, sep=',', parse_dates=['Date'], infer_datetime_format=True, index_col='Date')

    # sns.set_style('darkgrid')
    target = pd.DataFrame(data['Price'])
    # plt.figure(figsize=(15, 7))
    # plt.title('Gold Price')
    # plt.plot(target)

    target_tag1 = pd.DataFrame(target.iloc[0:1]).append(target.iloc[0:-1]).reset_index(drop=True)
    target_differ = pd.DataFrame(np.array(target) - np.array(target_tag1))

    target.columns = ['y']
    target_differ.columns = ['y']

    target_differ['differ'] = target_tag1

    for i in range(1, time_step + 1):
        target_differ['lag_{}'.format(i)] = target_differ.y.shift(i)

    data.drop(columns=['Price'], inplace=True)

    feature = series_to_supervised(data, time_step).reset_index(drop=True)

    data_differ = pd.concat([target_differ, feature], axis=1)

    data_differ = data_differ.dropna()

    hist = MyHist(target['y'], 100)
    # hist.plot_hist()

    hist = MyHist(target_differ['y'], 100)
    # hist.plot_hist()

    return data_differ.reset_index(drop=True)
