# -*-coding:utf-8-*-

"""
@Author: ethan
@Email: ethanwang279@gmail.com
@Datetime: 2020/2/21 15:57
@File: trainer.py
@Project: transformer_text_classification
@Description: None
"""

import torch
import config
from transformer.transformer_model import TransformerClassification

model = TransformerClassification(vocab_size=config.VOCAB_SIZE,
                                  max_seq_len=config.MAX_SEQ_LEN,
                                  num_category=config.NUM_CATEGORY,
                                  num_layers=config.NUM_LAYERS,
                                  d_model=config.DIM_MODEL,
                                  num_heads=config.NUM_HEADS,
                                  ffn_dim=config.DIM_FFN,
                                  dropout=config.DROPOUT)

optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()


def train():
    pass
