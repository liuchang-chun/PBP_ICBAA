import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import torch

import esm
from sklearn.decomposition import PCA


# # This article's path is shown as an example of C. jejuni.
# Define the file path
species_name = 'C.jejuni'

input_fasta_file = r"../data/{}/protein.test.all".format(species_name)
output_npy_file = r'../data/{}/numpy.test.esm.feature.npy'.format(species_name)
output_reduced_npy_file = r'../data/{}/jiangwei.numpy.test.esm.feature.npy'.format(species_name)

# Load the ESM model
pretrained_model_esm, alphabet_esm = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter_esm = alphabet_esm.get_batch_converter()
pretrained_model_esm.eval()


def get_1021_string(sequence, site=0):
    if len(sequence) > 1021:
        if site < 511:
            sequence_truncated = sequence[:1021]
            new_site = site
        elif site > len(sequence) - 511:
            sequence_truncated = sequence[-1021:]
            new_site = 1021 - (len(sequence) - site)
        else:
            sequence_truncated = sequence[site - 510: site + 510 + 1]
            new_site = 510
    else:
        new_site = site
        sequence_truncated = sequence
    return new_site, sequence_truncated

# Obtain ESM features
def get_esm2_3B_features(sequence):
    data = [("pid", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter_esm(data)
    with torch.no_grad():
        results = pretrained_model_esm(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    return token_representations[:, 1:-1, :][0]

# Extract features and save them
feature_representations = {}
for seq_record in tqdm(SeqIO.parse(input_fasta_file, "fasta")):
    prot_id = seq_record.id
    sequence = str(seq_record.seq)
    if len(sequence) > 1021:
        sequence = get_1021_string(sequence)[1]
    esm_all = get_esm2_3B_features(sequence)
    feature_representations[prot_id] = esm_all.numpy()

# Convert to numpy array
keys = sorted(feature_representations.keys())
num_keys = len(keys)
second_dim, third_dim = feature_representations[keys[0]].shape
three_d_array = np.empty((num_keys, second_dim, third_dim))
for i, key in enumerate(keys):
    three_d_array[i] = feature_representations[key]

# Save the original features
np.save(output_npy_file, three_d_array)
# Load the original feature and reduce the dimensionality
X1 = np.load(output_npy_file)

# Reshape your data to reduce dimensionality
X1_reshaped = X1.reshape(-1, X1.shape[-1])

# Create a PCA model and fit the data
pca = PCA(n_components=640)
X_pca = pca.fit_transform(X1_reshaped)

# Reshape back to the target shape
X_reduced = X_pca.reshape(X1.shape[0], X1.shape[1], 640)
print(X_reduced.shape)

# Save the data after dimensionality reduction
np.save(output_reduced_npy_file, X_reduced)