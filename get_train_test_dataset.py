# -*-coding:utf-8-*-

"""
@Author: ethan
@Email: ethanwang279@gmail.com
@Datetime: 2020/2/17 15:27
@File: get_train_test_dataset.py
@Project: transformer_text_classification
@Description: None
"""

import os
import pandas as pd
import math

root_dir = '/Users/ethan/THUCNews'
save_dir = "Dataset"


def get_dataset(samples_size):
    train_data = pd.DataFrame(columns=['category', 'item'])
    val_data, test_data= train_data.copy(), train_data.copy()
    for category in ["时尚", "时政", '科技']:
        dirname = os.path.join(root_dir, category)
        for _, di, files in os.walk(dirname):
            for j in range(samples_size):
                with open(os.path.join(dirname, files[j])) as f:
                    context = ""
                    for line in f.readlines():
                        context += line.strip('\r\n')
                    if j < math.ceil(samples_size * 0.8):
                        train_data.loc[len(train_data)] = [category, context]
                    elif math.ceil(samples_size * 0.8) < j < math.ceil(samples_size * 0.9):
                        val_data.loc[len(val_data)] = [category, context]
                    else:
                        test_data.loc[len(test_data)] = [category, context]
    return train_data, val_data, test_data


train, val, test = get_dataset(1000)
train.to_csv(os.path.join(save_dir, 'train_dataset.csv'), index=False)
val.to_csv(os.path.join(save_dir, 'val_dataset.csv'), index=False)
test.to_csv(os.path.join(save_dir, 'test_dataset.csv'), index=False)

