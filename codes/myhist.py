import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MyHist:
    def __init__(self, data, bins):
        self.start = np.array(data).min()
        self.end = np.array(data).max()
        self.width = (self.end - self.start) / bins
        self.total = data.shape[0]
        self.counts, self.bins = np.histogram(data, bins=bins)
        self.bin_num = bins

    def update(self, newdata):
        new_start = np.array(newdata).min()
        new_end = np.array(newdata).max()
        counts_temp = self.counts.tolist()
        bins_temp = self.bins.tolist()
        if new_start >= self.start:
            self.start = self.start
        else:
            for i in range(0, math.ceil((self.start - new_start) / self.width)):
                counts_temp.insert(0, 0)
                bins_temp.insert(0, self.start - (i + 1) * self.width)
                self.bin_num += 1
            # self.start = self.start - math.ceil((self.start - new_start) / self.width) * self.width
            self.start = bins_temp[0]
        if new_end <= self.end:
            self.end = self.end
        else:
            for i in range(0, math.ceil((new_end - self.end) / self.width)):
                counts_temp.append(0)
                bins_temp.append(self.end + (i + 1) * self.width)
                self.bin_num += 1
            # self.end = self.end + math.ceil((new_end - self.end) / self.width) * self.width
            self.end = bins_temp[-1]
        self.width = self.width
        self.total = self.total + newdata.shape[0]
        self.bins = np.array(bins_temp)
        # self.bin_num = math.ceil((self.end - self.start) / self.width)
        new_counts, new_bins = np.histogram(newdata, range=(self.start, self.end), bins=self.bin_num)
        counts_temp = np.array(counts_temp)
        self.counts = new_counts + counts_temp

    def clean(self):
        self.counts = np.zeros(self.counts.shape[0])

    def label_bins(self, data):
        new_start = np.array(data).min()
        new_end = np.array(data).max()
        counts_temp = self.counts.tolist()
        bins_temp = self.bins.tolist()
        new_bin_num = self.bin_num
        if new_start >= self.start:
            new_start = self.start
        else:
            # new_start = self.start - math.ceil((self.start - new_start) / self.width) * self.width
            for i in range(0, math.ceil((self.start - new_start) / self.width)):
                counts_temp.insert(0, 0)
                bins_temp.insert(0, self.start - (i + 1) * self.width)
                new_bin_num += 1
            new_start = bins_temp[0]
        if new_end <= self.end:
            new_end = self.end
        else:
            new_end = self.end + math.ceil((new_end - self.end) / self.width) * self.width
            for i in range(0, math.ceil((new_end - self.end) / self.width)):
                counts_temp.append(0)
                bins_temp.append(self.end + (i + 1) * self.width)
                new_bin_num += 1
            new_end = bins_temp[-1]
        # new_bin_num = math.ceil((new_end - new_start) / self.width)
        # new_bins = np.linspace(start=new_start, stop=new_end, num=new_bin_num + 1, endpoint=True)
        new_bins = np.array(bins_temp)
        # bin_label = []
        # for i in range(0, data.shape[0]):
        #     value = data.iloc[i]
        #     for j in range(0, new_bin_num):
        #         if new_bins[j] <= value < new_bins[j + 1]:
        #             bin_label.append(j)
        #             break
        #         if j == new_bin_num - 1:
        #             if value == new_bins[j + 1]:
        #                 bin_label.append(j)
        bin_label_mat = ((data - new_start) / self.width).astype(int)
        # m = (np.array(bin_label_mat) != np.array(bin_label))
        # a=pd.Series(bin_label_mat).iloc[m]
        # b=pd.Series(bin_label).iloc[m]
        # c = [bin_label_mat == new_bin_num]
        bin_label_mat.iloc[[bin_label_mat == new_bin_num]] = new_bin_num - 1
        # assert (np.array(bin_label_mat) == np.array(bin_label)).all()
        return pd.Series(bin_label_mat), pd.Series(counts_temp), new_bins

    def plot_hist(self):
        plt.figure(figsize=(4,4))
        plt.bar(range(self.bin_num), self.counts)
        plt.xticks([])
        plt.yticks([])

        plt.xlabel('Continuouis target value', fontproperties='Times New Roman', size=15)
        plt.ylabel('Number of samples', fontproperties='Times New Roman', size=15)
        plt.savefig('Gold_Price.svg', dpi=600, bbox_inches='tight', pad_inches=0)
        # plt.show()

    def compute_balance(self):
        balance = 1.0
        base = 1.0
        for i in range(0, self.bin_num):
            balance *= self.counts[i] * 1.0 / self.total
            base *= 1.0 / self.bin_num
        return 1.0 * balance / base


def label_bins_to(bins, data):
    # bin_label = []
    # for i in range(0, data.shape[0]):
    #     value = data.iloc[i]
    #     for j in range(0, len(bins) - 1):
    #         if bins[j] <= value < bins[j + 1]:
    #             bin_label.append(j)
    #             break
    #         if j == len(bins) - 1 - 1:
    #             if value == bins[j + 1]:
    #                 bin_label.append(j)
    bin_label_mat = ((data - bins[0]) / (bins[1] - bins[0])).astype(int)
    bin_label_mat.iloc[[bin_label_mat == len(bins) - 1]] = len(bins) - 2
    # assert (np.array(bin_label_mat) == np.array(bin_label)).all()
    return pd.Series(bin_label_mat)


def counts_sort(counts):
    return np.argsort(counts)
