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
                 training=True, shuffle=True, padding_batch=False):
        self.utt2npy = self._init_list(utt2npy)
        self.utt2target = self._init_dict(utt2target)
        self.targets_list = self._init_list(targets_list, col=0)
        self.target2int = dict()
        self.int2target = OrderedDict()
        self.utt2label_seq = self._init_dict(utt2label_seq, delimiter='\t')
        self.labels_list = self._init_list(labels_list, col=0)
        self.label2int = dict()
        self.training = training
        self.shuffle = shuffle
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

        if self.shuffle:
            random.shuffle(self.utt2npy)

    def _getitem_for_padding_batch(self, index):
        utt, npy = self.utt2npy[index]
        feat = np.load(npy)
        target = self.utt2target.get(utt, None)
        label_seq = self.utt2label_seq.get(utt, None)
        label_seq_length = len(label_seq) if isinstance(label_seq, np.ndarray) else 0
        sample = {"utt": utt, "feature": feat, "target": target, "length": feat.shape[0], 
                  "label_seq": label_seq, "label_seq_length": label_seq_length}
        return sample

    def _getitem_for_variable_length_batch(self, index_tlen):
        ind, truncated_len = index_tlen
        assert 0 <= ind and ind < self.dataset_size, "Invalid index \
                %d is out of range (0, %d)" % (ind, self.dataset_size)
        utt, npy = self.utt2npy[ind]
        feat = np.load(npy)
        target = self.utt2target.get(utt, None)

        if self.training:
            if feat.shape[0] <= truncated_len:
                # duplicate the short utterance
                feat = np.concatenate(
                    [feat] * (math.floor(truncated_len / feat.shape[0]) + 1), axis=0)
            idx = random.randrange(0, feat.shape[0] - truncated_len)
            feat = feat[idx:idx + truncated_len]

        if target != None:
            return [feat, target]
        return feat

    def get_int2target_dict(self):
        return self.int2target

    def __len__(self):
        return self.dataset_size

    def __targets_list__(self):
        return list(self.target2int.keys())

    def __getitem__(self, args):
        if self.padding_batch:
            sample = self._getitem_for_padding_batch(args)
        else:
            sample = self._getitem_for_variable_length_batch(args)
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
        self.batch_size = 1
        self._batch_size = batch_size
        # Note: DO NOT name `self._batch_size` as `self.batch_size`,
        # because `self.batch_size` is an attribute defined in the base class `DataLoader`,
        # and it is mutually exclusive with `batch_sampler`.
        self.training = training
        self.shuffle = shuffle
        self.fixed_len = fixed_len
        self.truncated_range = (truncated_min_len, truncated_max_len)
        self.padding_batch = padding_batch

        self.dataset = None
        self.dataset_size = 0
        self.batch_sampler = None
        self.collate_fn = None
        self._initial_data_loader()
        super(self.__class__, self).__init__(
            dataset=self.dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False)

    def _collate_fn(self, batch):
        assert len(batch) > 1
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
            label_seqs.append(torch.from_numpy(sample["label_seq"]))
            label_seq_lengths.append(sample["label_seq_length"])

        mask = torch.zeros((len(batch), max(lengths)))
        for i, l in enumerate(lengths):
            mask[i, :l] = 1

        utts = np.asarray(utts)
        feats = pad_sequence(feats, batch_first=True)
        targets = torch.from_numpy(np.asarray(targets))
        lengths = torch.from_numpy(np.asarray(lengths))
        label_seqs = pad_sequence(label_seqs, batch_first=True)
        label_seq_lengths = torch.from_numpy(np.asarray(label_seq_lengths))

        batch = {"utt": utts, "feats": feats, "targets": targets, "lengths": lengths, 
                "mask": mask, "label_seqs": label_seqs, "label_seq_lengths": label_seq_lengths}
        return batch

    def _batch_sampler(self):
        if self.training:
            for i in range(math.floor(self.dataset_size / self._batch_size)):
                # the remainder is droped.
                ind = np.arange(self._batch_size).reshape(
                    self._batch_size, 1) + i * self._batch_size
                if self.fixed_len > 0:
                    truncated_len = self.fixed_len
                else:
                    truncated_len = random.randrange(
                        self.truncated_range[0], self.truncated_range[1])
                tlen = truncated_len *  np.ones((self._batch_size, 1), dtype=np.int)
                yield np.concatenate((ind, tlen), axis=1)
        else:
            for i in range(self.dataset_size):
                # In the test stage, self._batch_size == 1
                ind = np.asarray([i], dtype=np.int).reshape(1, 1)
                yield np.concatenate((ind, np.ones((1, 1), dtype=np.int)), axis=1)

    def _initial_data_loader(self):
        self.dataset = SpeechDataset(
            self.utt2npy,
            utt2target=self.utt2target,
            targets_list=self.targets_list,
            utt2label_seq=self.utt2label_seq,
            labels_list=self.labels_list,
            training=self.training,
            shuffle=self.shuffle,
            padding_batch=self.padding_batch
        )
        self.dataset_size = self.dataset.__len__()
        if self.padding_batch:
            self.batch_sampler = None
            self.batch_size = self._batch_size
            self.collate_fn = self._collate_fn
        else:
            self.batch_sampler = self._batch_sampler()
            self.collate_fn = None
        # TODO: use collate_fn to prepare length-variable batch, 
        # then _batch_sampler() will be deprecated.

    def __len__(self):
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
                          training=True, padding_batch=False):

    data_loader = SpeechDataLoader(
        utt2npy, utt2target, targets_list, 
        utt2label_seq, labels_list,
        batch_size=batch_size, fixed_len=fixed_len, 
        num_workers=num_workers, training=training,
        padding_batch=padding_batch
    )

    dataset_size = data_loader.__len__()
    print('dataset_size: ', dataset_size)

    start = time.process_time()

    count = 0
    for i, batch in enumerate(data_loader):
        if padding_batch:
            for k, v in batch.items():
                print(k, type(v), v.shape)
            print("")
        else:
            print(batch[0])
            print(batch[1])
        count = i + 1
        if count == 10:
            break
    print("got n_batches: ", count)
    print("time elapsed: ", time.process_time() - start)


if __name__ == "__main__": 
    utt2npy = '../data/dev_utt2npy'
    utt2lang = '../data/dev_utt2lang'
    utt2phone_seq = '../data/dev_utt2phones_seq'
    languages = '../data/lang.list.txt'
    phones_list = '../data/phones.list.txt'

    # In the training stage, generate mini-batches data.
    data_loader_debugging(utt2npy, utt2lang, targets_list=languages, batch_size=6,
                      fixed_len=0, num_workers=4, training=True)
    print('-----' * 20)
    data_loader_debugging(utt2npy, utt2lang, languages, batch_size=6,
                      fixed_len=300, num_workers=4, training=True)
    print('-----' * 20)

    # In the test stage, one sample per test case
    data_loader_debugging(utt2npy, utt2lang, languages, batch_size=1,
                      num_workers=4, training=False)
    print('-----' * 20)

    data_loader_debugging(utt2npy, utt2lang, languages, batch_size=6,
                          utt2label_seq=utt2phone_seq, labels_list=phones_list,
                          num_workers=1, training=True, padding_batch=True)

