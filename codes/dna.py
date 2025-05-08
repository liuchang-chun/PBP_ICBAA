
import  numpy as np

index = 3
names = ['B.amyloliquefaciens', 'C.jejuni', 'C.pneumoniae', 'E.coli', 'H.pylori', 'L.interrogans', 'L.phytofermentans',
         'M.smegmatis','R.capsulatus','S.coelicolor','S.oneidensis','S.pyogenes','S.Typhimurium']
name = names[index]
Data_dir = f'../data/{name}'
#
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
    """Transcribe DNA sequences into RNA sequences"""
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
    """Read the list of DNA sequences from the file"""
    with open(file_path, 'r') as file:
        dna_sequences = [line.strip() for line in file.readlines()]
    return dna_sequences



def main():
    # Specify the file path that contains the DNA sequence.
    file_path = Data_dir + '/train.all.txt'

    dna_sequences = read_fasta(file_path)


    # Transcribe each DNA sequence into RNA sequence
    rna_sequences = [transcribe_dna_to_rna(dna) for dna in dna_sequences]

    # Open the output file in write mode.
    with open(Data_dir + '/train.rna.all.txt', 'w') as f:
        #
        for rna_sequence in rna_sequences:
            f.write(rna_sequence + '\n')
d = main()

