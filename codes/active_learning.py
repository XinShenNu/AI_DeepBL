import sys

from balancing_driven_AFs import *


def active_learning(sess, budget, original_indexes, new_data, true_label, balancing_AF, hist,
                    sampler_features, sampler_labels, once_budget=1):

    avail_indexes = np.array(original_indexes)

    print('------> Manual annotation')
    print('samples number -> ' + str(len(new_data)))
    print('budget of oracle -> ' + str(budget))
    print('')

    if balancing_AF == 'poor':
        annotated_sample_idx = poor(hist, sampler_features, sampler_labels, new_data, true_label, avail_indexes,
                                    budget)

    elif balancing_AF == 'weight':
        annotated_sample_idx = weight(hist, sampler_features, sampler_labels, new_data, true_label, avail_indexes,
                                      budget, once_budget)

    else:
        print('invalid balancing type')
        sys.exit(-1)

    return annotated_sample_idx
