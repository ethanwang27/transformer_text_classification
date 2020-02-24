# -*-coding:utf-8-*-

"""
@Author: ethan
@Email: ethanwang279@gmail.com
@Datetime: 2020/2/17 19:36
@File: process_file.py
@Project: transformer_text_classification
@Description: None
"""

import pandas as pd
import jieba
import re
from collections import Counter
import config

# 读取停用词
stopwords = []
with open(config.STOPWORDS_FILE, 'r') as f:
    stopwords = [word.strip('\n') for word in f.readlines()]


def seg_sentence(sentence, filter_stopwords):
    sentence = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", ' ', sentence.lower())
    sentence = re.sub(u"([0-9a-z]+)", ' ', sentence)
    seg_words = jieba.cut(sentence)
    result = ""
    if filter_stopwords:
        result = ' '.join([word for word in seg_words if word not in stopwords])
    else:
        result = ' '.join(seg_words)

    # 过滤多余的空格
    result = re.sub('[ ]+', ' ', result)
    return result


def proc_line(line, filter_stopwords):
    if isinstance(line, str):
        sentences = line.lower().split('|')
        seg_list = []
        for sentence in sentences:
            seg_list.append(seg_sentence(sentence, filter_stopwords))
        return ' '.join(seg_list)


def process_data():
    train_set = pd.read_csv(config.TRAIN_FILE)
    test_set = pd.read_csv(config.TEST_FILE)
    val_set = pd.read_csv(config.VAL_FILE)
    print('训练集数据总数：{} 测试集数据总数：{} 验证集数据总数：{}'.format(
        len(train_set), len(test_set), len(val_set)))

    train_set['item'] = train_set['item'].apply(proc_line, args={'filter_stopwords': config.IS_FILTER_STOPWORDS})
    test_set['item'] = test_set['item'].apply(proc_line, args={'filter_stopwords': config.IS_FILTER_STOPWORDS})
    val_set['item'] = val_set['item'].apply(proc_line, args={'filter_stopwords': config.IS_FILTER_STOPWORDS})

    train_set.to_csv(config.TRAIN_SEG_FILE, index=False, encoding='utf-8')
    test_set.to_csv(config.TEST_SEG_FILE, index=False, encoding='utf-8')
    val_set.to_csv(config.VAL_SEG_FILE, index=False, encoding='utf-8')

    vocab = [config.UNKNOWN_TOKEN, config.PAD_TOKEN, config.START_TOKEN, config.END_TOKEN]
    [vocab.append(word) for line in train_set['item'] for word in line.split()]
    [vocab.append(word) for line in test_set['item'] for word in line.split()]
    [vocab.append(word) for line in val_set['item'] for word in line.split()]


if __name__ == '__main__':
    process_data()