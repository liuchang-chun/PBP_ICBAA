
import  numpy as np

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

def transcribe_dna_to_rna(dna_sequence):
    """将DNA序列转录成RNA序列"""
    rna_sequence = ""
    for nucleotide in dna_sequence:
        if nucleotide == 'A':
            rna_sequence += 'U'
        elif nucleotide == 'T':
            rna_sequence += 'A'
        elif nucleotide == 'C':
            rna_sequence += 'G'
        elif nucleotide == 'G':
            rna_sequence += 'C'
        else:
            raise ValueError(f"Invalid nucleotide: {nucleotide}")
    return rna_sequence


def read_dna_sequences_from_file(file_path):
    """从文件中读取DNA序列列表"""
    with open(file_path, 'r') as file:
        dna_sequences = [line.strip() for line in file.readlines()]
    return dna_sequences


#  Transcription of DNA sequence into RNA sequence
def main():

    file_path = './data/C.jejuni/test'

    dna_sequences = read_fasta(file_path)


    rna_sequences = [transcribe_dna_to_rna(dna) for dna in dna_sequences]


    with open('./data/C.jejuni/test.rna.all.txt', 'w') as f:

        for rna_sequence in rna_sequences:
            f.write(rna_sequence + '\n')

"""
if __name__ == "__main__":
    main()"""

import numpy as np
from sklearn.model_selection import train_test_split

coden_dict = {'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',                             # alanine<A>
              'UGU': 'C', 'UGC': 'C',                                                 # systeine<C>
              'GAU': 'D', 'GAC': 'D',                                                 # aspartic acid<D>
              'GAA': 'E', 'GAG': 'E',                                                 # glutamic acid<E>
              'UUU': 'F', 'UUC': 'F',                                                 # phenylanaline<F>
              'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',                             # glycine<G>
              'CAU': 'H', 'CAC': 'H',                                                 # histidine<H>
              'AUU': 'I', 'AUC': 'I', 'AUA': 'I',                                       # isoleucine<I>
              'AAA': 'K', 'AAG': 'K',                                                 # lycine<K>
              'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',         # leucine<L>
              'AUG': 'M',                                                          # methionine<M>
              'AAU': 'N', 'AAC':'N',                                               # asparagine<N>
              'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',                         # proline<P>
              'CAA': 'Q', 'CAG': 'Q',                                               # glutamine<Q>
              'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',   # arginine<R>
              'UCU': 'S', 'UCC':  'S', 'UCA':  'S', 'UCG': 'S', 'AGU': 'S', 'AGC':'S',   # serine<S>
              'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',                         # threonine<T>
              'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',                         # valine<V>
              'UGG': 'W',                                                          # tryptophan<W>
              'UAU': 'Y', 'UAC': 'Y',                                               # tyrosine(Y)
              'UAA': 'Z', 'UAG': 'Z', 'UGA': 'Z',                                    # STOP code
              }

# the amino acid code adapting 21-dimensional vector (20 amino acid and 1 STOP code)




def coden_1(seq):
    # 创建一个空字符串用于存储蛋白质序列
    protein_seq = ""

    # 遍历RNA序列，从第一个到倒数第三个核苷酸，每次跳过一个核苷酸
    for i in range(0, len(seq) - 2, 3):
        # 根据coden_dict字典获取当前密码子对应的氨基酸
        codon = seq[i:i + 3]  # 将RNA中的'T'替换为'U'
        amino_acid = coden_dict_1.get(codon, None)
        if amino_acid == '_':
            break
        else:
            protein_seq += amino_acid
        # 获取密码子对应的氨基酸




    return protein_seq


# 定义RNA到蛋白质的字典映射
coden_dict_1 = {
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',  # Alanine (A)
    'UGU': 'C', 'UGC': 'C',  # Cysteine (C)
    'GAU': 'D', 'GAC': 'D',  # Aspartic acid (D)
    'GAA': 'E', 'GAG': 'E',  # Glutamic acid (E)
    'UUU': 'F', 'UUC': 'F',  # Phenylalanine (F)
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',  # Glycine (G)
    'CAU': 'H', 'CAC': 'H',  # Histidine (H)
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I',  # Isoleucine (I)
    'AAA': 'K', 'AAG': 'K',  # Lysine (K)
    'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',  # Leucine (L)
    'AUG': 'M',  # Methionine (M)
    'AAU': 'N', 'AAC': 'N',  # Asparagine (N)
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',  # Proline (P)
    'CAA': 'Q', 'CAG': 'Q',  # Glutamine (Q)
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',  # Arginine (R)
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S',  # Serine (S)
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',  # Threonine (T)
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',  # Valine (V)
    'UGG': 'W',  # Tryptophan (W)
    'UAU': 'Y', 'UAC': 'Y',  # Tyrosine (Y)
    'UAA': 'Y', 'UAG': 'Y', 'UGA': 'W',  # STOP codons
}


#  The translation of RNA into protein
with open('C:/Users/22750/Desktop/iEnhancer-DCSV-main/data/C.jejuni/test.rna.all.txt', 'r') as f:

    with open('C:/Users/22750/Desktop/iEnhancer-DCSV-main/data/C.jejuni/protein.test.all', 'w') as output_file:
        with open('C:/Users/22750/Desktop/iEnhancer-DCSV-main/data/C.jejuni/new.test.rna.all', 'w') as output_file_rna:
            # 逐行读取RNA序列
            index = 1
            for line in f:




                line = line.strip()

                protein_seq = coden_1(line)




                output_file_rna.write(f">{index}\n{line}\n")  # 添加fasta格式标识符
                output_file.write(f">{index}\n{protein_seq}\n")  # 添加fasta格式标识符
                index += 1

