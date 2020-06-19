#! /usr/bin/env python3

import sys
import os
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from collections import OrderedDict


class CTC_LID_Model(nn.Module):

    def __init__(self, config, use_cuda=True):
        super(CTC_LID_Model, self).__init__()
        self.config = config
        self.input_size = config.feat_dim
        self.hidden_size = config.hidden_size
        self.bn_size = config.bn_size
        self.output_size = config.num_classes
        self.num_ctc_classes = config.num_ctc_classes
        self.ctc_loss_weight = config.ctc_loss_weight

        self.ctc_loss = nn.CTCLoss()
        self.nll_loss = nn.CrossEntropyLoss()

        self.lstm_layers = nn.LSTM(self.input_size, self.hidden_size, 
                                   num_layers=config.num_rnn_layers, 
                                   batch_first=True, 
                                   bidirectional=config.bidirectional
                                   )
        self.lstm_layers.flatten_parameters()

        lstm_output_size = self.hidden_size
        if config.bidirectional:
            lstm_output_size *= 2
        
        self.dropout = nn.Dropout(p=config.dropout_rate)

        # feed forward layers for bottleneck feature
        self.ffn_layers = nn.Sequential()
        self.ffn_layers.add_module('linear_layer_0', nn.Linear(lstm_output_size, self.bn_size))
        self.ffn_layers.add_module('prelu', nn.PReLU())
        self.ffn_layers.add_module('linear_layer_1', nn.Linear(self.bn_size, self.bn_size))

        self.output_layer_for_ctc = nn.Linear(self.bn_size, self.num_ctc_classes)
        self.output_layer_for_lid = nn.Linear(self.bn_size, self.output_size)

        self.model = None
        self.use_cuda = use_cuda

    def forward(self, feats, mask):
        batch_size, length, feat_dim = feats.size()

        lstm_output, last_hidden_status = self.lstm_layers(feats)

        if self.config.do_train:
            lstm_output = self.dropout(lstm_output)

        bn_embd = self.ffn_layers(lstm_output)

        if self.config.do_train or self.config.pretrain_ctc_model:
            ctc_logits = self.output_layer_for_ctc(bn_embd) 
        else:
            ctc_logits = None

        lid_logits = self.output_layer_for_lid(bn_embd)
        mask = mask.contiguous().view(batch_size, length, -1).expand(batch_size, length, lid_logits.size(2))
        lid_logits = lid_logits * mask
        lid_logits = lid_logits.sum(dim=1) / mask.sum(dim=1)
        
        return (ctc_logits, lid_logits)

    def compute_loss(self, lid_logits, inputs_length, lid_targets, ctc_logits, ctc_labels, ctc_labels_length, pretrain_ctc_model=False):
        # calculate CTC loss, transpose logitc from (batch_size, length, C) to (length, batch_size, C)
        ctc_logits = ctc_logits.transpose(0, 1).contiguous()
        loss1 = self.ctc_loss(ctc_logits, ctc_labels.int(), inputs_length.int(), ctc_labels_length.int())

        if pretrain_ctc_model:
            return (None, loss1, None)

        loss2 = self.nll_loss(lid_logits, lid_targets)

        total_loss = self.ctc_loss_weight * loss1 + loss2
        return (total_loss, loss1, loss2)

    def evaluate(self, logits, targets):
        _, pred = logits.max(dim=1)
        correct = pred.eq(targets).float()
        acc = correct.sum().item() 
        return (pred, acc)

    def predict(self, lid_logits):
        _, pred = lid_logits.max(dim=1)
        return pred

