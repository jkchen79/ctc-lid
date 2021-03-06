#! /usr/bin/env python3

import numpy as np
import sys
import os

import kaldiio


def read_file_linebyline(filename):
    with open(filename) as fp:
        data = fp.read().splitlines()
    return data


def write_file(filename, data, mode='w'):
    with open(filename, mode) as fp:
        fp.write("%s\n" % "\n".join(data))

if len(sys.argv) < 2:
    print("Usage: python kaldi_ark2npy.py <feats.scp> [feats_npy_dir]\n")
    exit(1)

kaldi_feats_scp = sys.argv[1]

assert os.path.isfile(kaldi_feats_scp),  "Invalid path to feats.scp file"

data_base = os.path.abspath(os.path.dirname(kaldi_feats_scp))

if 2 == len(sys.argv):
    feats_npy_dir = os.path.join(data_base, 'feats_npy')
else:
    feats_npy_dir = sys.argv[2]

if not os.path.exists(feats_npy_dir):
    os.mkdir(feats_npy_dir)

utt2npy = []

feats = kaldiio.load_scp(kaldi_feats_scp)

for utt, feat in feats.items():
    feat_out = os.path.join(feats_npy_dir, "%s.npy" % utt)
    utt2npy.append("%s %s" % (utt, feat_out))
    np.save(feat_out, feat)
    #  print(feat.shape)

write_file(os.path.join(data_base, 'utt2npy'), utt2npy)
