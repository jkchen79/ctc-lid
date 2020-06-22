#! /usr/bin/env python3
# encoding=utf8

# Copyright 2018  Sun Yat-sen University (author: Jinkun Chen)

import sys
import os
import random
import math
import time
import numpy as np

from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

"""
Data loader of speech processing
"""


def read_file_linebyline(filename, encoding=None):
    with open(filename) as rf:
        data = [line.strip() for line in rf]
    data = [line for line in data if len(line) > 0]
    return data


def write_file(filename, data, mode='w'):
    with open(filename, mode) as wf:
        wf.write("%s" % "\n".join(data))


def shuffle_file(filename):
    lines = read_file_linebyline(filename)
    random.shuffle(lines)
    write_file(filename, lines)


class SpeechDataset(Dataset):

    def __init__(self, utt2npy, utt2target=None, targets_list=None,
                 utt2label_seq=None, labels_list=None,
                 training=True, padding_batch=False):
        self.utt2npy = self._init_list(utt2npy)
        self.utt2target = self._init_dict(utt2target)
        self.targets_list = self._init_list(targets_list, col=0)
        self.target2int = dict()
        self.int2target = OrderedDict()
        self.utt2label_seq = self._init_dict(utt2label_seq, delimiter='\t')
        self.labels_list = self._init_list(labels_list, col=0)
        self.label2int = dict()
        self.training = training
        self.padding_batch = padding_batch
        self.dataset_size = len(self.utt2npy)
        self._init_dataset()

    def _read_file(self, filename):
        if not os.path.isfile(filename):
            raise ValueError("the file %s does not exist!" % filename)
        data = []
        with open(filename) as fp:
            for line in fp:
                line = line.strip() 
                if len(line) > 0:
                    data.append(line)
        return data

    def _init_list(self, filename, col=None, delimiter=None):
        if filename is None:
            return []
        data = self._read_file(filename)
        if col != None and col >= 0:
            xlist = []
            for line in data:
                fields = line.split(delimiter)
                if len(fields) > col:
                    xlist.append(fields[col])
            return xlist
        return data

    def _init_dict(self, filename, delimiter=None):
        if filename is None:
            return dict()
        data = self._read_file(filename)
        xdict = dict()
        for line in data:
            fields = line.split(delimiter)
            if len(fields) == 2:
                xdict[fields[0]] = fields[1]
        return xdict

    def _init_dataset(self):
        self.utt2npy = [line.split() for line in self.utt2npy]
        targets = sorted(set(self.targets_list))
        if len(targets) == 0:
            targets = sorted(set(self.utt2target.values()))
        self.target2int = dict(zip(targets, range(len(targets))))
        self.int2target = dict(zip(range(len(targets)), targets))

        for utt, target in self.utt2target.items():
            self.utt2target[utt] = self.target2int.get(target, 0)

        uniq_labels = OrderedDict(zip(self.labels_list, range(len(self.labels_list))))
        self.labels_list = list(uniq_labels.keys())
        self.label2int = dict(zip(self.labels_list, range(len(self.labels_list))))
        for utt, label_seq in self.utt2label_seq.items():
            label_seq = np.asarray(list(map(lambda t:self.label2int.get(t, 0), label_seq.split()[1:-1])))
            self.utt2label_seq[utt] = label_seq
        
        if self.training:
            assert self.utt2target is not None, "utt2target must be provided in training phase! "

    def get_int2target_dict(self):
        return self.int2target

    def __len__(self):
        return self.dataset_size

    def __targets_list__(self):
        return list(self.target2int.keys())

    def __getitem__(self, index):
        utt, npy = self.utt2npy[index]
        feat = np.load(npy)
        target = self.utt2target.get(utt)
        sample = {"utt": utt, "feature": feat, "target": target, "length": feat.shape[0]}
        label_seq = self.utt2label_seq.get(utt, None)
        label_seq_length = len(label_seq) if isinstance(label_seq, np.ndarray) else 0
        if label_seq is not None:
            sample["label_seq"] = label_seq
            sample["label_seq_length"] = label_seq_length
        return sample


class SpeechDataLoader(DataLoader):

    """
    return the data in mini-batch, with type of torch.Tensor and size of:
        feats:   [batch_size, truncated_len, feature_size]
        targets: [batch_size]

    """

    def __init__(self, utt2npy, utt2target=None, targets_list=None, 
                 utt2label_seq=None, labels_list=None, batch_size=1,
                 num_workers=1, training=True, shuffle=True,
                 fixed_len=0, truncated_min_len=200, truncated_max_len=1024,
                 padding_batch=False):
        self.utt2npy = utt2npy
        self.utt2target = utt2target
        self.targets_list = targets_list
        self.utt2label_seq = utt2label_seq
        self.labels_list = labels_list
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.training = training
        self.shuffle = shuffle
        self.fixed_len = fixed_len
        self.truncated_range = (truncated_min_len, truncated_max_len)
        self.padding_batch = padding_batch

        self.dataset = None
        self.dataset_size = 0
        self.batch_sampler = None
        self._initial_data_loader()
        super(self.__class__, self).__init__(
            dataset=self.dataset,
            collate_fn=self._collate_fn,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=False)

    def _pad_batch_samples(self, batch):
        utts = []
        feats = []
        targets = []
        lengths = []
        label_seqs = []
        label_seq_lengths = []
        for sample in batch:
            utts.append(sample["utt"])
            feats.append(torch.from_numpy(sample["feature"]))
            targets.append(sample["target"])
            lengths.append(sample["length"])
            label_seq = sample.get("label_seq", None)
            if label_seq is not None:
                label_seqs.append(torch.from_numpy(label_seq))
                label_seq_lengths.append(sample["label_seq_length"])

        mask = torch.zeros((len(batch), max(lengths)))
        for i, l in enumerate(lengths):
            mask[i, :l] = 1

        utts = np.asarray(utts)
        feats = pad_sequence(feats, batch_first=True)
        targets = torch.from_numpy(np.asarray(targets))
        lengths = torch.from_numpy(np.asarray(lengths))

        batch = {"utt": utts, "feats": feats, "targets": targets, "lengths": lengths, "mask": mask}

        if len(label_seqs) > 0:
            label_seqs = pad_sequence(label_seqs, batch_first=True)
            label_seq_lengths = torch.from_numpy(np.asarray(label_seq_lengths))
            batch["label_seqs"] = label_seqs
            batch["label_seq_lengths"] = label_seq_lengths

        return batch

    def _get_variable_length_batch(self, batch):
        if self.training:
            if self.fixed_len > 0:
                truncated_len = self.fixed_len
            else:
                truncated_len = random.randrange(self.truncated_range[0], self.truncated_range[1])

            utts = []
            feats = []
            targets = []
            lengths = []
            for sample in batch:
                utts.append(sample["utt"])
                targets.append(sample["target"])
                lengths.append(truncated_len)
                feat = sample["feature"]
                if feat.shape[0] <= truncated_len:
                    # duplicate the short utterance
                    feat = np.concatenate([feat] * (math.floor(truncated_len / feat.shape[0]) + 1), axis=0)
                idx = random.randrange(0, feat.shape[0] - truncated_len)
                feat = feat[idx:idx + truncated_len]
                feats.append(feat)

            utts = np.asarray(utts)
            feats = torch.from_numpy(np.asarray(feats))
            targets = torch.from_numpy(np.asarray(targets))
            lengths = torch.from_numpy(np.asarray(lengths))
            batch = {"utt": utts, "feats": feats, "targets": targets, "lengths": lengths}
        else:
            batch = self._pad_batch_samples(batch)
        return batch

    def _collate_fn(self, batch):
        if self.padding_batch:
            batch = self._pad_batch_samples(batch)
        else:
            batch = self._get_variable_length_batch(batch)
        return batch

    def _initial_data_loader(self):
        self.dataset = SpeechDataset(
            self.utt2npy,
            utt2target=self.utt2target,
            targets_list=self.targets_list,
            utt2label_seq=self.utt2label_seq,
            labels_list=self.labels_list,
            training=self.training,
            padding_batch=self.padding_batch
        )
        self.dataset_size = self.dataset.__len__()

    def __len__(self):
        return self.dataset_size

    def get_dataset_size(self):
        return self.dataset_size

    def __targets_list__(self):
        return self.dataset.__targets_list__()

    def get_int2target_dict(self):
        if self.dataset != None:
            return self.dataset.get_int2target_dict()
        return OrderedDict()

def data_loader_debugging(utt2npy, utt2target=None, targets_list=None,
                          utt2label_seq=None, labels_list=None,
                          batch_size=64, fixed_len=0, num_workers=4, 
                          training=True, shuffle=True, padding_batch=False):

    data_loader = SpeechDataLoader(utt2npy, utt2target, targets_list, 
                                   utt2label_seq, labels_list,
                                   batch_size=batch_size, fixed_len=fixed_len, 
                                   num_workers=num_workers, training=training,
                                   shuffle=shuffle, padding_batch=padding_batch
                                   )

    dataset_size = data_loader.get_dataset_size()
    print('dataset_size: ', dataset_size)

    num_batchs = int(dataset_size / batch_size)
    print("num_batchs: ", n_batchs)

    start = time.process_time()

    for i, batch in enumerate(data_loader):
        for k, v in batch.items():
            print(k, type(v), v.shape)
        print("")
        if i + 1 == 10:
            break
    print("got num_batchs: ", i + 1)
    print("time elapsed: ", time.process_time() - start)


if __name__ == "__main__": 
    utt2npy = '../data/dev_utt2npy'
    utt2lang = '../data/dev_lang_label.txt'
    utt2phones_seq = '../data/dev_utt2phones_seq'
    languages = '../data/lang.list.txt'
    phones_list = '../data/phones.list.txt'

    In the training stage, generate mini-batches data.
    print("variable length batch")
    data_loader_debugging(utt2npy, utt2lang, targets_list=languages, batch_size=64,
                      fixed_len=0, num_workers=4, training=True, shuffle=False)
    print('-----' * 20)

    print("truncate fixed length batch")
    data_loader_debugging(utt2npy, utt2lang, languages, batch_size=64,
                      fixed_len=1024, num_workers=4, training=True, shuffle=True)
    print('-----' * 20)

    print("test mode")
    # In the test stage, one sample per test case
    data_loader_debugging(utt2npy, utt2lang, languages, batch_size=1,
                      num_workers=4, training=False)
    print('-----' * 20)

    print("prepare phones sequences for ctc")
    data_loader_debugging(utt2npy, utt2lang, languages, batch_size=64,
                          utt2label_seq=utt2phones_seq, labels_list=phones_list,
                          num_workers=1, training=True, padding_batch=True)

