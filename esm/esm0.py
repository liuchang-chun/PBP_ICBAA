"""
      Author  : Suresh Pokharel
      Email   : sureshp@mtu.edu
"""
from Bio import SeqIO

"""
import required libraries
"""
import numpy as np
import pandas as pd
#from Bio import SeqIO
from keras import backend as K
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import torch
from transformers import T5EncoderModel, T5Tokenizer, Trainer, TrainingArguments, EvalPrediction
import re
import gc
import esm
from functools import partial
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn import metrics
from scipy import stats
import esm
"""
define file paths and other parameters
"""
# C:\Users\22750\Desktop\iEnhancer-DCSV-main\data\protein\123

# output_csv_file = r"D:\project\LM-OGlcNAc-Site-main\data(after)\H\results.csv"


cutoff_threshold_ankh = 0.496
cutoff_threshold_esm = 0.797
cutoff_threshold_prott5 = 0.669

# create results dataframe
results_df = pd.DataFrame(columns = ['prot_desc', 'position','site_residue', 'ankh_prob(Th = 0.496)','prot_t5_prob(Th = 0.669)','esm2_prob(Th = 0.797)', 'final_prediction'])


# Load ESM-2 model
pretrained_model_esm, alphabet_esm = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter_esm = alphabet_esm.get_batch_converter()
pretrained_model_esm.eval()  # disables dropout for deterministic results


# 将给定的序列进行处理，以便返回一个长度为1021的子序列，同时更新位置参数
def get_1021_string(sequence, site=0):
    """
        We are taking one sequence at a time because of the memory issue, this can be improved a lot
    """

    # truncate sequence to peptide of 1024 if it is greater
    if len(sequence) > 1021:
        if site < 511:
            # take first 1023 window
            sequence_truncated = sequence[:1021]
            new_site = site

        elif site > len(sequence) - 511:
            sequence_truncated = sequence[-1021:]
            new_site = 1021 - (len(sequence) - site)
        else:
            # Use new position just to extract the feature, store original
            sequence_truncated = sequence[site - 510: site + 510 + 1]
            new_site = 510
    else:
        # change nothing
        new_site = site
        sequence_truncated = sequence

    return new_site, sequence_truncated


def get_esm2_3B_features(sequence):
    # prepare input df in the format that model accepts
    # data = [
    #     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    # ]

    # prepare dataframe of truncated string
    data = [
        ("pid", sequence),
    ]

    batch_labels, batch_strs, batch_tokens = batch_converter_esm(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = pretrained_model_esm(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # return only residue level embeddings so that we can treat them exactly as prott5 features that we are already using
    return token_representations[:, 1:-1, :][0]


feature_representations = {}

# 假设 get_esm2_3B_features 是一个函数，用于获取给定序列的ESM特征

# C:\Users\22750\Desktop\iEnhancer-DCSV-main\data\C.jejuni\protein.test.all
input_fasta_file = r"./data/C.jejuni/protein.test.all" # load test sequence

for seq_record in tqdm(SeqIO.parse(input_fasta_file, "fasta")):
    prot_id = seq_record.id
    sequence = str(seq_record.seq)

    # if sequence is longer than 1021, truncate
    if len(sequence) > 1021:
        sequence = get_1021_string(sequence)

    # 为每个序列计算ESM特征
    esm_all = get_esm2_3B_features(sequence)

    # 存储特征表示，键为蛋白ID，值为特征
    feature_representations[prot_id] = esm_all

    # 现在您可以根据需要使用 feature_representations[prot_id] 中的特征
    # 例如，获取特定位置的ESM特征：
    for index, amino_acid in enumerate(sequence):
        if amino_acid in ['S', 'T']:
            site = index + 1  # 考虑索引从0开始
            # 获取特定位置的ESM特征
            X_test_esm = esm_all[site - 1]  # 注意这里的索引是从1开始的，所以需要做相应的调整
            # 接下来您可以根据需要处理 X_test_esm

np.save(r'./data/C.jejuni/esm.test.feature.npy', feature_representations)





