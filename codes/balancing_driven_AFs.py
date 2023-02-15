from __future__ import division
import random
from collections import Counter

from myhist import *

random.seed(0)


def poor(hist, sampler_features, sampler_labels, candidate_features, candidate_labels, avail_indexes, al_budget):

    candidate_features = candidate_features.reset_index(drop=True)
    candidate_labels = candidate_labels.reset_index(drop=True)

    bin_label, counts, new_bins = hist.label_bins(candidate_labels)
    # sort = counts_sort(counts)

    sampler_bin_label = label_bins_to(new_bins, sampler_labels)

    selected_idx = []
    label_features_dict = {}
    label_mean_feat_dict = {}
    label_count_dict = Counter(sampler_bin_label)  # 统计范例中每个bin_label的计数信息

    # 统计每个sampler_bin_label的features，生成字典
    # (此处需改进，速度较慢)
    # for i in range(0, sampler_bin_label.shape[0]):
    for label in label_count_dict.keys():
        # label = sampler_bin_label.iloc[i]
        # features = sampler_features.iloc[i]
        label_features_dict[label] = np.array(sampler_features.iloc[np.array(sampler_bin_label) == label])
        # label_features_dict[label] = np.array(sampler_features.iloc[[k for k in range(0, sampler_bin_label.shape[0])
        #                                                              if sampler_bin_label.iloc[k] == label]])
        # if label not in label_features_dict.keys():
        #     label_features_dict[label] = np.array(features)
        # else:
        #     label_features_dict[label] = np.vstack((label_features_dict[label], np.array(features)))

    print('before')
    print(str(str(label_count_dict)))

    # #################go to balancing mode ###################################################

    print("start balancing:")

    selected = 0

    minors = []  # 少数bin
    majors = []  # 多数bin
    # sampler_bin_label的计数均值
    mean_count = math.floor(np.mean(np.array([label_count_dict[k] for k in label_count_dict.keys()])))

    for label in label_count_dict.keys():

        # 根据sampler_bin_label的计数和均值将其分为少数bin和多数bin两部分
        if label_count_dict[label] < mean_count:
            minors.append(label)
        else:
            majors.append(label)

        # mean features: 属于每个sampler_bin_label的features的均值
        if label_count_dict[label] > 1:
            label_mean_feat_dict[label] = np.mean(label_features_dict[label], axis=0)
        else:
            label_mean_feat_dict[label] = label_features_dict[label]

    while selected < al_budget:
        # get the count of label with minimum number of samplers
        rarest_count = min(label_count_dict.values())
        rarest_label = [label for label in label_count_dict.keys() if label_count_dict[label] == rarest_count]

        # for label in label_count_dict.keys():
        #     if label_count_dict[label] < rarest_count:
        #         rarest_count = label_count_dict[label]

        # (此处可优化：上一步直接得到rarest_count的label集合，这里只对集合中的label做循环)
        for label in rarest_label:
            # assert (label_count_dict[label] == rarest_count)
            # min_minus_max_majors = []
            # (此处需优化：将for循环改为矩阵运算，否则时间随候选集合大小增长)
            # for i in range(0, candidate_features.shape[0]):
            # feature = candidate_features.iloc[i]
            # 当前候选特征和当前bin_label的特征均值的二范数
            dist_label = np.linalg.norm(candidate_features - np.array(label_mean_feat_dict[label]), axis=1)
            # assert (dist_label >= 0)
            # 当前候选特征和每一个多数bin_label的特征均值的二范数的最小值
            dist_maj_label = [np.linalg.norm(np.array(candidate_features - np.array(label_mean_feat_dict[maj_label])),
                                             axis=1) for maj_label in majors]
            min_dist_maj_label = np.min(np.array(dist_maj_label), axis=0)
            # 要使此式子的值最小，则候选特征和当前拥有最少样本的bin_label的特征均值的二范数最小，
            # 候选特征和每一个多数bin_label的特征均值的二范数的最小值最大
            min_minus_max_majors = dist_label - min_dist_maj_label

            # 远离任何一个多数bin，接近当前最少样本的bin
            # (可改进的地方：若新bin样本离最少数bin远，被选择的机会就小于接近最少数bin的样本)
            best_idx = np.argmin(np.array(min_minus_max_majors))
            if selected < al_budget:
                # get the true class of the selected index to simulate a human annotation
                true_label = bin_label.iloc[best_idx]
                true_feature = candidate_features.iloc[best_idx]

                selected += 1

                # 将选择的样本的bin_label和特征加入字典中
                if true_label not in label_count_dict.keys():
                    label_count_dict[true_label] = 1
                    label_features_dict[true_label] = np.array(true_feature)
                    label_mean_feat_dict[true_label] = label_features_dict[true_label]
                else:
                    label_count_dict[true_label] += 1
                    label_features_dict[true_label] = np.vstack(
                        (label_features_dict[true_label], np.array(true_feature)))
                    label_mean_feat_dict[true_label] = np.mean(label_features_dict[true_label], axis=0)

                # 将选中的样本从候选样本中删除
                candidate_features = candidate_features.drop(best_idx, axis=0).reset_index(drop=True)
                candidate_labels = candidate_labels.drop(best_idx, axis=0).reset_index(drop=True)
                bin_label = bin_label.drop(best_idx, axis=0).reset_index(drop=True)

                # 将选中的样本索引加入返回列表，并从可选索引中删除
                selected_idx.append(avail_indexes[best_idx])
                avail_indexes = np.delete(avail_indexes, best_idx, axis=0)

                # 重新划分多数bin和少数bin
                minors = []
                majors = []
                mean_count = math.floor(np.mean(np.array([label_count_dict[k] for k in label_count_dict.keys()])))

                for key in label_count_dict.keys():
                    if label_count_dict[key] < mean_count:
                        minors.append(key)
                    else:
                        majors.append(key)

            #######################################

            elif selected >= al_budget:
                break
            else:
                print('error')

    print('after')
    print(str(label_count_dict))

    print("stopped balancing:")
    assert (selected == len(selected_idx))
    return selected_idx


def weight(hist, sampler_features, sampler_labels, candidate_features, candidate_labels,
           avail_indexes, al_budget, once_budget):
    candidate_features = candidate_features.reset_index(drop=True)
    candidate_labels = candidate_labels.reset_index(drop=True)

    bin_label, counts, new_bins = hist.label_bins(candidate_labels)
    # sort = counts_sort(counts)

    sampler_bin_label = label_bins_to(new_bins, sampler_labels)

    selected_idx = []
    label_features_dict = {}
    label_mean_feat_dict = {}
    label_count_dict = Counter(sampler_bin_label)  # 统计范例中每个bin_label的计数信息

    # 统计每个sampler_bin_label的features，生成字典
    for label in label_count_dict.keys():
        label_features_dict[label] = np.array(sampler_features.iloc[np.array(sampler_bin_label) == label])

    print('before')
    print(str(str(label_count_dict)))

    # ##################go to balancing mode ###################################################

    print("start balancing:")

    selected = 0

    minors = []  # 少数bin
    majors = []  # 多数bin
    # sampler_bin_label的计数均值
    mean_count = math.floor(np.mean(np.array([label_count_dict[k] for k in label_count_dict.keys()])))

    for label in label_count_dict.keys():

        # 根据sampler_bin_label的计数和均值将其分为少数bin和多数bin两部分
        if label_count_dict[label] < mean_count:
            minors.append(label)
        else:
            majors.append(label)

        # mean features: 属于每个sampler_bin_label的features的均值
        if label_count_dict[label] > 1:
            label_mean_feat_dict[label] = np.mean(label_features_dict[label], axis=0)
        else:
            label_mean_feat_dict[label] = label_features_dict[label]

    while selected < al_budget:
        # get the count of label with minimum number of samplers
        rarest_count = min(label_count_dict.values())
        rarest_labels = [label for label in label_count_dict.keys() if label_count_dict[label] == rarest_count]

        for rarest_label in rarest_labels:
            dist_rarest_label = np.linalg.norm(
                np.array(candidate_features - np.array(label_mean_feat_dict[rarest_label])), axis=1)
            # 当前候选特征和当前rarest_labels的特征均值的二范数的最小值
            # dist_rarest_label = [
            #     np.linalg.norm(np.array(candidate_features - np.array(label_mean_feat_dict[rarest_label])),
            #                    axis=1) for rarest_label in rarest_labels]
            # min_dist_rarest_label = np.min(np.array(dist_rarest_label), axis=0)
            # 当前候选特征和每一个多数maj_label的特征均值的二范数的最小值
            dist_maj_label = [np.linalg.norm(np.array(candidate_features - np.array(label_mean_feat_dict[maj_label])),
                                             axis=1) for maj_label in majors]
            min_dist_maj_label = np.min(np.array(dist_maj_label), axis=0)
            # 要使此式子的值最小，则候选特征和当前rarest_label的特征均值的二范数最小值最小，
            # 候选特征和每一个maj_label的特征均值的二范数的最小值最大
            dist_balance = dist_rarest_label - min_dist_maj_label
            # # 改进：避免若新bin样本离最少数bin远导致被选择的机会就小于接近最少数bin的样本，增加了pattern_special项
            # dist_pattern_special = [np.linalg.norm(np.array(candidate_features - np.array(
            # label_mean_feat_dict[label])),axis=1) for label in label_mean_feat_dict.keys()]
            # min_dist_pattern_special = np.min(np.array(dist_pattern_special), axis=0)
            # dist_pattern_similar = np.e ** np.array(min_dist_pattern_special * -1.0 /
            #                                         (np.mean(np.array(dist_pattern_special), axis=0)))
            # weight_balance_special = dist_balance + 0.3*dist_pattern_similar
            # # weight_balance_special = dist_balance/min_dist_pattern_special

            # best_idx = np.argmin(np.array(weight_balance_special))
            # best_idx = np.argpartition(np.array(dist_balance), once_budget)[:once_budget]
            best_idx = np.array(dist_balance).argsort()[:once_budget]

            if selected < al_budget:
                # get the true class of the selected index to simulate a human annotation
                true_labels = bin_label.iloc[best_idx].reset_index(drop=True)
                true_features = candidate_features.iloc[best_idx].reset_index(drop=True)

                selected += once_budget

                for i in range(0, once_budget):
                    # 将选择的样本的bin_label和特征加入字典中
                    true_label = true_labels.iloc[i]
                    true_feature = true_features.iloc[i]
                    if true_label not in label_count_dict.keys():
                        label_count_dict[true_label] = 1
                        label_features_dict[true_label] = np.array(true_feature)
                        label_mean_feat_dict[true_label] = label_features_dict[true_label]
                    else:
                        label_count_dict[true_label] += 1
                        label_features_dict[true_label] = np.vstack(
                            (label_features_dict[true_label], np.array(true_feature)))
                        label_mean_feat_dict[true_label] = np.mean(label_features_dict[true_label], axis=0)

                # 将选中的样本从候选样本中删除
                candidate_features = candidate_features.drop(best_idx, axis=0).reset_index(drop=True)
                candidate_labels = candidate_labels.drop(best_idx, axis=0).reset_index(drop=True)
                bin_label = bin_label.drop(best_idx, axis=0).reset_index(drop=True)

                # 将选中的样本索引加入返回列表，并从可选索引中删除
                selected_idx = np.append(selected_idx, avail_indexes[best_idx])
                avail_indexes = np.delete(avail_indexes, best_idx, axis=0)

                # 重新划分多数bin和少数bin
                minors = []
                majors = []
                mean_count = math.floor(np.mean(np.array([label_count_dict[k] for k in label_count_dict.keys()])))

                for key in label_count_dict.keys():
                    if label_count_dict[key] < mean_count:
                        minors.append(key)
                    else:
                        majors.append(key)

            #######################################

            elif selected >= al_budget:
                break
            else:
                print('error')

    print('after')
    print(str(label_count_dict))

    print("stopped balancing:")
    assert (selected == len(selected_idx))
    return selected_idx
