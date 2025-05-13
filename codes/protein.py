import numpy as np
from sklearn.model_selection import train_test_split


index = 3
names = ['B.amyloliquefaciens', 'C.jejuni', 'C.pneumoniae', 'E.coli', 'H.pylori', 'L.interrogans', 'L.phytofermentans',
         'M.smegmatis','R.capsulatus','S.coelicolor','S.oneidensis','S.pyogenes','S.Typhimurium']
name = names[index]
Data_dir = f'../data/{name}'

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

# the amino acid code adapting 21-dimensional vector (20 amino acid and 1 STOP code

def coden_1(seq):
    # Create an empty string to store the protein sequence
    protein_seq = ""

    # Traverse the RNA sequence, skipping one nucleotide at a time from the first to the third-to-last nucleotide
    for i in range(0, len(seq) - 2, 3):
        # Obtain the amino acid corresponding to the current codon according to the coden_dict dictionary
        codon = seq[i:i + 3]  # Replace 'T' with 'U' in RNA
        amino_acid = coden_dict_1.get(codon, None)
        if amino_acid == '_':
            break
        else:
            protein_seq += amino_acid


    return protein_seq


# Define RNA-to-protein dictionary mappings
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



# Open the file containing RNA sequences.
with open(Data_dir + '/train.rna.all.txt', 'r') as f:
    # Open a new file to store the protein sequences that match the criteria
    with open(Data_dir + '/protein.train.all', 'w') as output_file:
        with open(Data_dir + '/new.train.rna.all', 'w') as output_file_rna:

            index = 1
            for line in f:



                # Remove the newline character at the end of the line.
                line = line.strip()
                # If the current row is not empty, the RNA sequence is processed
                protein_seq = coden_1(line)

                        # Write the protein sequence that satisfies the condition to a new file
                output_file_rna.write(f">{index}\n{line}\n")  # Add a FASTA format identifier
                output_file.write(f">{index}\n{protein_seq}\n")
                index += 1

