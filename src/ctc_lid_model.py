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

from resnet import ResNet, ResidualBlock

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

        self.front_net = nn.Sequential()

        resnet18 = ResNet(ResidualBlock, [2, 2, 2, 2], [1, 1, 2, 2])
        self.front_net.add_module('resnet18', resnet18)
        front_output_size = 1280

        self.lstm_layers_for_ctc = nn.LSTM(front_output_size, self.hidden_size, 
                                   num_layers=config.num_rnn_layers, 
                                   batch_first=True, 
                                   bidirectional=config.bidirectional
                                   )
        self.lstm_layers_for_ctc.flatten_parameters()

        self.lstm_layers_for_lid = nn.LSTM(front_output_size, self.hidden_size, 
                                   num_layers=config.num_rnn_layers, 
                                   batch_first=True, 
                                   bidirectional=config.bidirectional
                                   )
        self.lstm_layers_for_lid.flatten_parameters()

        lstm_output_size = self.config.hidden_size
        if config.bidirectional:
            lstm_output_size *= 2

        #  self.dropout = nn.Dropout(p=config.dropout_rate)
        # feed forward layers for bottleneck feature

        self.ffn_layers_for_ctc = nn.Sequential()
        self.ffn_layers_for_ctc.add_module('linear_layer_0', nn.Linear(lstm_output_size, self.bn_size))
        self.ffn_layers_for_ctc.add_module('prelu', nn.PReLU())
        self.ffn_layers_for_ctc.add_module('linear_layer_1', nn.Linear(self.bn_size, self.num_ctc_classes))

        self.ffn_layers_for_lid = nn.Sequential()
        self.ffn_layers_for_lid.add_module('linear_layer_0', nn.Linear(lstm_output_size, self.bn_size))
        self.ffn_layers_for_lid.add_module('prelu', nn.PReLU())
        self.ffn_layers_for_lid.add_module('linear_layer_1', nn.Linear(self.bn_size, self.output_size))

        self.model = None
        self.use_cuda = use_cuda

    def forward(self, feats, mask):
        batch_size, length, feat_dim = feats.size()
        # for CNN, reshape the feats to shape of (batch_size, channel, length, width), channel == 1.
        feats = torch.unsqueeze(feats, 1)
        front_output = self.front_net(feats)
        front_output = torch.transpose(front_output, 1, 2)
        shape = front_output.shape
        # reshape the CNN output to shape of (batch_size, length_out, channel_out x width_out).
        front_output = front_output.contiguous().view(shape[0], shape[1], -1)
        mask = mask[:, ::4]

        if self.config.do_train or self.config.pretrain_ctc_model:
            lstm_output_ctc, _ = self.lstm_layers_for_ctc(front_output)
            ctc_logits = self.ffn_layers_for_ctc(lstm_output_ctc) 
        else:
            ctc_logits = None

        lstm_output_lid, _ = self.lstm_layers_for_lid(front_output)
        lid_logits = self.ffn_layers_for_lid(lstm_output_lid)
        mask = mask.unsqueeze(2)
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

