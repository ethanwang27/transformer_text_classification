# -*-coding:utf-8-*-

"""
@Author: ethan
@Email: ethanwang279@gmail.com
@Datetime: 2020/2/21 16:05
@File: config.py
@Project: transformer_text_classification
@Description: None
"""

# 数据集路径
TRAIN_FILE = './Dataset/train_dataset.csv'
TEST_FILE = './Dataset/test_dataset.csv'
VAL_FILE = './Dataset/val_dataset.csv'

TRAIN_SEG_FILE = './Dataset/train_seg_dataset.csv'
TEST_SEG_FILE = './Dataset/test_seg_dataset.csv'
VAL_SEG_FILE = './Dataset/val_seg_dataset.csv'

VOCAB_FILE = './Dataset/vocab.txt'

STOPWORDS_FILE = './Dataset/stopwords.txt'

# Word Embedding
PAD_TOKEN, UNKNOWN_TOKEN = '<PAD>', '<UNK>'
START_TOKEN, END_TOKEN = '<SOS>', '<EOS>'

# 模型参数
VOCAB_SIZE = 512
MAX_SEQ_LEN = 340
DIM_MODEL = 512
NUM_LAYERS = 8
NUM_HEADS = 8


DROPOUT = 0.8
DIM_FFN = 2048
NUM_CATEGORY = 3

# 训练参数
IS_FILTER_STOPWORDS = True
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 1e-5
