
import numpy as np
import pandas as pd
#from Bio import SeqIO
#
input_fasta_file = "./data/E.coli/esm.test.feature.npy" # load test sequence
train = np.load(input_fasta_file, allow_pickle=True).item()


keys = sorted(train.keys())
num_keys = len(keys)
second_dim, third_dim = train[keys[0]].shape
three_d_array = np.empty((num_keys, second_dim, third_dim))
for i, key in enumerate(keys):
    three_d_array[i] = train [key]
np.save(r'./data/E.coli/numpy.test.esm.feature.npy', three_d_array)


