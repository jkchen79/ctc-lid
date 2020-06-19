#! /usr/bin/env python3
# Copyright 2018  Sun Yat-sen University (author: Jinkun Chen)

import sys
import argparse
import json
import time
import os

class Options_Parser:

    def __init__(self, description=""):
        self.parser = argparse.ArgumentParser(description=description)
        self._dataset_options()
        self._dataloader_options()
        self._model_options()
        self._optimizer_options()
        self._running_options()

    def _dataset_options(self):
        self.parser.add_argument(
            "--train-utt2npy", dest="train_utt2npy", type=str, default=None,
            help="The path of the utt2npy file for training.")
        self.parser.add_argument(
            "--train-utt2target", dest="train_utt2target", type=str, default=None,
            help="The path of the utt2target file for training.")
        self.parser.add_argument(
            "--train-utt2label-seq", dest="train_utt2label_seq", type=str, default=None,
            help="The path of the utt2label_seq file for training.")
        self.parser.add_argument(
            "--eval-utt2npy", dest="eval_utt2npy", type=str, default=None,
            help="The path of the utt2npy file for evaluation.")
        self.parser.add_argument(
            "--eval-utt2target", dest="eval_utt2target", type=str, default=None,
            help="The path of the utt2target file for evaluation.")
        self.parser.add_argument(
            "--eval-utt2label-seq", dest="eval_utt2label_seq", type=str, default=None,
            help="The path of the utt2label_seq file for evaluation.")
        self.parser.add_argument(
            "--targets-list", dest="targets_list", type=str, default=None,
            help="The path of the targets_list file.")
        self.parser.add_argument(
            "--labels-list", dest="labels_list", type=str, default=None,
            help="The path of the labels_list file.")

    def _dataloader_options(self):
        self.parser.add_argument(
            '--epochs', dest='epochs', type=int, default=90,
            help='The maximum epochs for training phase.')
        self.parser.add_argument(
            '--batch-size', dest='batch_size', type=int, default=64,
            help='Batch size for DataLoader in training phase.')
        self.parser.add_argument(
            '--fixed-length', dest='fixed_length', type=int, default=0,
            help='If fixed_length > 0, the data loader generates feature sequences in fixed length.')
        self.parser.add_argument(
            '--truncated-min-len', dest='truncated_min_len', type=int, default=200,
            help='the lower bound of the variable truncated range.')
        self.parser.add_argument(
            '--truncated-max-len', dest='truncated_max_len', type=int, default=1024,
            help='the upper bound of the variable truncated range.')
        self.parser.add_argument(
            '--padding-batch', dest='padding_batch', action='store_true',
            help='If True, pad sequence to the max length in a batch.')
        self.parser.add_argument(
            '--num-workers', dest='num_workers', type=int, default=10,
            help='the number of the workers to load data parallelly.')

    def _optimizer_options(self):
        self.parser.add_argument(
            "--optimizer", dest="optimizer", type=str, default="adam",
            help="The type of optimizer.")
        self.parser.add_argument(
            "--lr", dest="lr", type=float, default=0.001,
            help="The initial learning rate.")
        self.parser.add_argument(
            '--ctc-loss-weight', dest='ctc_loss_weight', type=float, default=0.3,
            help='The weight of CTC Loss.')
        self.parser.add_argument(
            "--pretrain-ctc-lr", dest="pretrain_ctc_lr", type=float, default=0.001,
            help="The initial learning rate for pre-train CTC model.")

    def _model_options(self):
        self.parser.add_argument(
            '--feat-dim', dest='feat_dim', type=int, default=40,
            help='The dimension of acustic features.')
        self.parser.add_argument(
            '--hidden-size', dest='hidden_size', type=int, default=1024,
            help='The hidden size of the recurrent layers.')
        self.parser.add_argument(
            '--num-rnn-layers', dest='num_rnn_layers', type=int, default=1,
            help='The number of recurrent layers.')
        self.parser.add_argument(
            '--bidirectional', dest='bidirectional', action='store_true',
            help='If True, use a bidirectional recurrent layer.')
        self.parser.add_argument(
            '--dropout-rate', dest='dropout_rate', type=float, default=0.1,
            help="The dropout rate for lstm_layers' output.")
        self.parser.add_argument(
            '--bn-size', dest='bn_size', type=int, default=256,
            help='The size of bottleneck representation.')
        self.parser.add_argument(
            '--num-classes', dest='num_classes', type=int, default=10,
            help='The number of classes (targets).')
        self.parser.add_argument(
            '--num-ctc-classes', dest='num_ctc_classes', type=int, default=180,
            help='The number of CTC classes (targets).')

    def _running_options(self):
        self.parser.add_argument(
            "--ckpt-dir", dest="ckpt_dir", type=str, default="../ckpt",
            help="the directory to save checkpoint files.")
        self.parser.add_argument(
            "--ckpt", dest="ckpt", type=str, default="",
            help="the path of the checkpoint file to be restored.")
        self.parser.add_argument(
            '--do-train', dest='do_train', action='store_true',
            help='If True, run the CTC-LID model in training mode.')
        self.parser.add_argument(
            '--do-eval', dest='do_eval', action='store_true',
            help='If True, run the CTC-LID model in evaluation mode.')
        self.parser.add_argument(
            '--do-predict', dest='do_predict', action='store_true',
            help='If True, run the CTC-LID model in prediction mode.')
        self.parser.add_argument(
            "--pretrain-ctc-epochs", dest="pretrain_ctc_epochs", type=int, default=0,
            help="The number of epochs to pre-train CTC model.")
        self.parser.add_argument(
            '--freeze-parameters', dest='freeze_parameters', action='store_true',
            help='If True, freeze the pre-trained parameters.')
        self.parser.add_argument(
            '--restore-from-ckpt', dest='restore_from_ckpt', action='store_true',
            help='If True, restore the model from a checkpont file.')
        self.parser.add_argument(
            "--log-file", dest="log_file", type=str, default="",
            help="The path of logging file for the running job.")

    def parse_args(self):
        config = self.parser.parse_args()
        mode = ""
        if config.do_train:
            mode = 'train'
        elif config.do_eval:
            mode = 'test'
        elif config.do_predict:
            mode = 'predict'

        config.pretrain_ctc_model = config.pretrain_ctc_epochs > 0
        timestamp = time.strftime("%m%d-%H%M%S", time.localtime())
        config.ckpt_dir = config.ckpt_dir.rstrip('/')
        if config.do_train and os.path.basename(config.ckpt_dir) == 'ckpt':
            config.ckpt_dir = os.path.join(config.ckpt_dir, "job_%s_ctc_lid_%s" % (mode, timestamp))

        if len(mode) > 0 and len(config.log_file) == 0:
            config.log_file = "./log/job_%s_ctc_lid_%s.log.txt" % (mode, timestamp)

        if config.restore_from_ckpt or config.do_eval or config.do_predict:
            assert os.path.isfile(config.ckpt) and os.path.exists(config.ckpt)

        if config.freeze_parameters:
            config.ctc_loss_weight = 0.0

        if config.do_eval or config.do_predict:
            config.dropout_rate = 0
        return config


    def register(self, arg, dest, arg_type, default=None, action=None, help=""):
        if isinstance(arg_type, bool) and action is not None:
            self.parser.add_argument(
                arg, dest=dest, action=action, help=help)
        else:
            self.parser.add_argument(
                arg, dest=dest, type=arg_type, default=default, help=help)


if __name__ == '__main__':
    parser = Options_Parser("debug")
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
