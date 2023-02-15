import time
import pandas as pd
import numpy as np


def save_to_excel(filename, predict_value, mae, mse, rmse, mape, smape, r2, train_time, test_time):
    table = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape, 'R2': r2, 'training_time': train_time,
             'testing_time': test_time}

    index = 0
    predict_mat = np.array(predict_value).T
    for array in predict_mat:
        table['y{}'.format(index)] = array
        index += 1

    df = pd.DataFrame(table)
    df.to_excel(filename)
    print('结果已保存至：'+filename)
