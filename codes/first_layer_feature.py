import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

np.random.seed(0)
tf.random.set_seed(1)  # for reproducibility


def read_fasta(fasta_file_name):
    seqs = []
    seqs_num = 0
    file = open(fasta_file_name)

    for line in file.readlines():
        if line.strip() == '':
            continue

        if line.startswith('>'):
            seqs_num = seqs_num + 1
            continue
        else:
            seq = line.strip()

            result1 = 'N' in seq
            result2 = 'n' in seq
            if result1 == False and result2 == False:
                seqs.append(seq)
    return seqs

def one_hot(sequence):
    num = len(sequence)
    length = len(sequence[0])
    feature = np.zeros((num, length, 4))
    for i in range(num):
        for j in range(length):
            if sequence[i][j] == 'A':
                feature[i, j] = [1, 0, 0, 0]
            elif sequence[i][j] == 'T':
                feature[i, j] = [0, 1, 0, 0]
            elif sequence[i][j] == 'C':
                feature[i, j] = [0, 0, 1, 0]
            elif sequence[i][j] == 'G':
                feature[i, j] = [0, 0, 0, 1]
    return feature

def to_one_hot(seqs):
    base_dict = {
        'a': 0, 'c': 1, 'g': 2, 't': 3,
        'A': 0, 'C': 1, 'G': 2, 'T': 3
    }

    one_hot_4_seqs = []
    for seq in seqs:

        one_hot_matrix = np.zeros([4, len(seq)], dtype=float)
        index = 0
        for seq_base in seq:
            one_hot_matrix[base_dict[seq_base], index] = 1
            index = index + 1

        one_hot_4_seqs.append(one_hot_matrix)
    return one_hot_4_seqs

# NCP coding
def to_properties_code(seqs):
    properties_code_dict = {
        'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1],
        'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([len(seq), 3], dtype=float)
        m = 0
        for seq_base in seq:
            properties_matrix[m, :] = properties_code_dict[seq_base]
            m = m + 1
        properties_code.append(properties_matrix)
    return properties_code

def to_properties_code_1(seqs):
    properties_code_dict = {
        'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1],
        'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([3, len(seq)], dtype=float)
        m = 0
        for seq_base in seq:
            properties_matrix[:, m] = properties_code_dict[seq_base]
            m = m + 1
        properties_code.append(properties_matrix)
    return properties_code

# NCPD
def to_properties_code_NCPD(seqs):
    properties_code_dict = {
        'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1],
        'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]
    }

    properties_code = []
    for seq in seqs:
        seq_length = len(seq)
        # Initialization is a chemical property plus a density property
        properties_matrix = np.zeros([seq_length, 4], dtype=float)
        # Calculate the density
        density = {base: seq.count(base) / seq_length for base in 'ACGTacgt'}
        m = 0
        for seq_base in seq:
            # Add chemical properties
            properties_matrix[m, :3] = properties_code_dict[seq_base]
            # Add a density attribute
            properties_matrix[m, 3] = density[seq_base]
            m += 1
        properties_code.append(properties_matrix)

    return properties_code

def to_properties_density_code(seqs):
    properties_code_dict = {
        'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1],
        'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([4, len(seq)], dtype=float)
        A_num = 0
        C_num = 0
        G_num = 0
        T_num = 0
        All_num = 0
        for seq_base in seq:
            if seq_base == "A":
                All_num += 1
                A_num += 1
                Density = A_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
            if seq_base == "C":
                All_num += 1
                C_num += 1
                Density = C_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
            if seq_base == "G":
                All_num += 1
                G_num += 1
                Density = G_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
            if seq_base == "T":
                All_num += 1
                T_num += 1
                Density = T_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
        properties_code.append(properties_matrix)
    return properties_code

def Binary(seqs):
    properties_code_dict = {
        'A': [0, 0], 'C': [0, 1], 'G': [1, 0], 'T': [1, 1],
        'a': [0, 0], 'c': [0, 1], 'g': [1, 0], 't': [1, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([len(seq), 2], dtype=float)
        m = 0
        for seq_base in seq:
            properties_matrix[m, :] = properties_code_dict[seq_base]
            m = m + 1
        properties_code.append(properties_matrix)
    return properties_code

def to_C2_code(seqs):
    properties_code_dict = {
        'A': [0, 0], 'C': [1, 1], 'G': [1, 0], 'T': [0, 1],
        'a': [0, 0], 'c': [1, 1], 'g': [1, 0], 't': [0, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([2, len(seq)], dtype=float)
        m = 0
        for seq_base in seq:
            properties_matrix[:, m] = properties_code_dict[seq_base]
            m = m + 1
        properties_code.append(properties_matrix)
    return properties_code
