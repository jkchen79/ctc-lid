#! /usr/bin/env python3

import sys
import os
import argparse
import json
import re
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from collections import OrderedDict

from options_parser import Options_Parser
from speech_dataloader import SpeechDataLoader
from ctc_lid_model import CTC_LID_Model

import warnings
warnings.filterwarnings('ignore')


def mkdir(dir_path):
    dir_path = dir_path.rstrip('/')
    dirs_to_be_created = []
    while not os.path.exists(dir_path):
        dirs_to_be_created.append(dir_path)
        dir_path = os.path.dirname(dir_path)
    for d in reversed(dirs_to_be_created):
        os.mkdir(d)

def get_logger(log_file):
    log_dir = os.path.dirname(log_file)
    mkdir(log_dir)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def predict(config, logger):
    pass

def do_evaluation(model, dataloader, logger, pretrain_ctc_model=False):
    num_samples = 0
    correct_cnt = 0
    ctc_loss = 0
    int2target = dataloader.get_int2target_dict()
    for i, batch_data in enumerate(dataloader):
        utt = batch_data["utt"]
        feats = batch_data["feats"].cuda()
        lid_targets = batch_data["targets"].cuda()
        lengths = batch_data["lengths"].cuda()
        mask = batch_data["mask"].cuda()
        label_seqs = batch_data["label_seqs"].cuda()
        label_seq_lengths = batch_data["label_seq_lengths"].cuda()
        num_samples += feats.size(0)
        ctc_logits, lid_logits = model(feats, mask)
        if pretrain_ctc_model:
            _, loss, _ = model.module.compute_loss(lid_logits, lengths, 
                    lid_targets, ctc_logits, label_seqs, label_seq_lengths)
            ctc_loss += loss
            logger.info("dev batch %d, ctc_loss %.6f" % (i, loss))
        else:
            pred, acc = model.module.evaluate(lid_logits, lid_targets)
            for u, t, p in zip(utt, lid_targets.cpu().numpy(), pred.cpu().numpy()):
                logger.info(" ".join((u, int2target[t], int2target[p])))
            correct_cnt += acc 

    acc_rate = float(correct_cnt) / num_samples
    avg_ctc_loss = ctc_loss / num_samples
    if not pretrain_ctc_model:
        logger.info("acc = %d / %d" % (correct_cnt, num_samples))
    return (acc_rate, avg_ctc_loss)

def evaluate(config, logger):
    model = CTC_LID_Model(config)
    ckpt_params = OrderedDict()
    for k, v in torch.load(config.ckpt).items():
        k = re.sub("^module.", "", k)
        ckpt_params[k] = v
    model.load_state_dict(ckpt_params)
    model.eval()
    model = nn.DataParallel(model)
    model = model.cuda()

    eval_dataloader = SpeechDataLoader(config.eval_utt2npy,
                                       utt2target=config.eval_utt2target,
                                       targets_list=config.targets_list,
                                       utt2label_seq=config.eval_utt2label_seq,
                                       labels_list=config.labels_list,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       training=False,
                                       shuffle=False,
                                       padding_batch=config.padding_batch
                                       )
    acc_rate, _ = do_evaluation(model, eval_dataloader, logger, config.pretrain_ctc_model)
    logger.info("evaluation acc rate = %f" % acc_rate)

def setup_optimizer(model, optimizer_name, learning_rate):
    optimizer = None
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(trainable_parameters, 
                        lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    else:
        raise ValueError
    return optimizer

def train_epochs(model, config, pretrain_ctc_model=False):
    if pretrain_ctc_model:
        logger.info("Begin to pre-train the CTC model ...")
        num_epochs = config.pretrain_ctc_epochs
        learning_rate = config.pretrain_ctc_lr
    else:
        logger.info("Begin to fine-tuning ...")
        num_epochs = config.epochs
        learning_rate = config.lr

    optimizer = setup_optimizer(model, config.optimizer, learning_rate)
    logger.info(optimizer)

    for epoch in range(num_epochs):
        if epoch == int(num_epochs * 0.4) or epoch == int(num_epochs * 0.75):
            learning_rate /= 10
            logger.info("learning_rate decays to %f" % learning_rate)
            optimizer = setup_optimizer(model, config.optimizer, learning_rate)
            logger.info(optimizer)

        training_dataloader = SpeechDataLoader(
                                      config.train_utt2npy,
                                      utt2target=config.train_utt2target,
                                      targets_list=config.targets_list,
                                      utt2label_seq=config.train_utt2label_seq,
                                      labels_list=config.labels_list,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      training=True,
                                      shuffle=True,
                                      padding_batch=config.padding_batch
                                      )

        dev_dataloader = SpeechDataLoader(config.eval_utt2npy,
                                       utt2target=config.eval_utt2target,
                                       targets_list=config.targets_list,
                                       utt2label_seq=config.eval_utt2label_seq,
                                       labels_list=config.labels_list,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       training=False,
                                       shuffle=False,
                                       padding_batch=config.padding_batch
                                       )

        num_batchs = training_dataloader.get_dataset_size() / config.batch_size

        total_ctc_loss = 0
        for i, batch_data in enumerate(training_dataloader):
            feats = batch_data["feats"].cuda()
            lid_targets = batch_data["targets"].cuda()
            lengths = batch_data["lengths"].cuda()
            mask = batch_data["mask"].cuda()
            label_seqs = batch_data["label_seqs"].cuda()
            label_seq_lengths = batch_data["label_seq_lengths"].cuda()

            # Forward + Backward + Optimize
            ctc_logits, lid_logits = model(feats, mask)

            optimizer.zero_grad()
            if pretrain_ctc_model:
                _, ctc_loss, _ = model.module.compute_loss(lid_logits, lengths,
                    lid_targets, ctc_logits, label_seqs, label_seq_lengths, pretrain_ctc_model)
                loss = ctc_loss
                total_ctc_loss += ctc_loss
                logger.info("Epoch [%d/%d], Iter [%d/%d], CTCLoss %.4f, Length [%d]" %
                       (epoch, num_epochs, i, num_batchs, ctc_loss.data, feats.size()[1]))
            else:
                #  loss, l1, l2 = model.compute_loss(lid_logits, lengths,
                loss, l1, l2 = model.module.compute_loss(lid_logits, lengths,
                    lid_targets, ctc_logits, label_seqs, label_seq_lengths, pretrain_ctc_model)
                total_ctc_loss += l1
                logger.info("Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, CTCLoss %.4f, "
                        "LIDLoss: %.4f, Length [%d]" % (epoch, num_epochs, i, num_batchs, 
                                                loss.data, l1.data, l2.data, feats.size()[1]))
            loss.backward()
            optimizer.step()

        if pretrain_ctc_model:
            avg_ctc_loss = total_ctc_loss / num_batchs
            logger.info("Epoch [%d/%d], CTC avg_ctc_loss = %.6f" % (epoch, num_epochs, avg_ctc_loss))
            ckpt_path = "%s/ckpt-pretrain-ctc-model-epoch-%d-avg-ctc-loss-%.6f.mdl" % (config.ckpt_dir, epoch, avg_ctc_loss)
        else:
            acc, _ = do_evaluation(model, dev_dataloader, logger)
            logger.info("Epoch [%d/%d], dev acc = %.6f" % (epoch, num_epochs, acc))
            ckpt_path = "%s/ckpt-epoch-%d-acc-%.4f.mdl" % (config.ckpt_dir, epoch, acc)
        torch.save(model.state_dict(), ckpt_path)


def train(config, logger):
    torch.manual_seed(997)

    model = CTC_LID_Model(config)

    if config.restore_from_ckpt:
        ckpt_params = OrderedDict()
        for k, v in torch.load(config.ckpt).items():
            k = re.sub("^module.", "", k)
            ckpt_params[k] = v
        model.load_state_dict(ckpt_params)

    model = nn.DataParallel(model)
    model = model.cuda()
    logger.info("nnet model:")
    logger.info(model)

    mkdir(config.ckpt_dir)

    # pre-train the CTC model 
    if config.pretrain_ctc_model:
        train_epochs(model, config, config.pretrain_ctc_model)

    if config.freeze_parameters:
        for name, param in model.named_parameters():
            if "lstm_layers" in name or "ffn_layers" in name:
                logger.info("freeze parameter: %s, size of %s" % (name, str(param.size())))
                param.requires_grad = False

    # fine-tune
    train_epochs(model, config)


if __name__ == '__main__':
    description = 'Args Parser.'
    parser = Options_Parser(description)
    config = parser.parse_args()

    logger = get_logger(config.log_file)
    logger.info("config:")
    logger.info(json.dumps(vars(config), indent=4))

    if config.do_train:
        train(config, logger)
    elif config.do_eval:
        evaluate(config, logger)
    elif config.do_predict:
        predict(config, logger)
