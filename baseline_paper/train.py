"""
Unofficial re-implementation (from the paper text only) of:
  Imtiaz & Khan, "Enhanced cross-dataset EEG-based emotion recognition using
  unsupervised domain adaptation", Computers in Biology and Medicine 184 (2025) 109394.

Pipeline: pre-training (group DRO + classifier discrepancy) -> cluster computation
-> domain adaptation with GPTDS -> inference with PC-TTA.

Values the paper does NOT specify are marked  # ASSUMPTION  and exposed as arguments.
Usage:
  python train.py --source deap --target seed --deap_dir DEAP/data_preprocessed_python \
                  --seed_dir SEED/Preprocessed_EEG
  python train.py --source seed --target deap ...
"""

import argparse, glob, json, os, pickle, time
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, resample
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"
# ////////////////////////////////////////////////////////
BANDS = {"delta": (1, 3), "theta": (4, 7), "alpha": (8, 13), "beta": (14, 30), "gamma": (31, 50)}
DEAP_CH = ["FP1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1",
           "OZ", "PZ", "FP2", "AF4", "FZ", "F4", "F8", "FC6", "FC2", "CZ", "C4", "T8", "CP6", "CP2",
           "P4", "P8", "PO4", "O2"]
SEED_CH = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
           "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ",
           "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7",
           "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
           "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
SEED_TRIAL_LABELS = None  # read from label.mat

def make_windows(x, fs, win=2, step=1):
    L, S = int(win * fs), int(step * fs)
    n = (x.shape[1] - L) // S + 1
    return np.stack([x[:, i * S:i * S + L] for i in range(n)])  # (n, C, L)

def psd_features(w, fs):
    L = w.shape[-1]
    X = np.fft.rfft(w.astype(np.float32), axis=-1)
    psd = (np.abs(X) ** 2) / L
    f = np.fft.rfftfreq(L, 1 / fs)
    df = f[1] - f[0]
    out = [psd[..., (f >= lo) & (f <= hi)].sum(-1) * df for lo, hi in BANDS.values()]
    return np.stack(out, 1).reshape(len(w), -1).astype(np.float32)