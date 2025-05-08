Here's the English translation of the provided document description:

---

# File Description

- `first_layer_main.py`
  The main program of the model, which requires input data paths and their quantities.

- `first_layer_feature.py`
  Contains the features of the model.

- `data`
  Contains the model's data, including promoter positive and negative samples, as well as protein files saved as `.npy` files.

- `esm`
  - `esm_main.py`
    ESM feature encoding.




## example
- For example, this paper uses the species *C. jejuni* as an example.
  - Step 1: The `data` directory contains the promoter positive and negative samples for *C. jejuni*. The input format in the paper is `Data_dir + '/train_p.txt'`, where `Data_dir = f'./data/{name}'`, and `name` is *C. jejuni*.
  - Step 2: We obtain the ESM features for *C. jejuni*. We transform each sequence into a protein form, considering each sequence as a special matrix without regard to whether the DNA sequence can be transcribed into a protein. 
  - In this step, we store each species's protein in the `protein.fasta` format in the `data` directory, and input the protein sequences into `esm_main.py` to obtain the ESM feature matrix for each sequence. 
  -At this point, we have obtained the ESM feature matrix.
  - Step 3: We input the obtained feature matrix and sample data into the `first_layer_main.py` main program for training.
-- Warning: In this paper, the ESM feature extraction will be shown using the C. jejuni species as an example, and the name of the pathway species can be changed only when changing other species
-- Warning: The paths in main.py of the first layer are demonstrated using C. jejuni as an example. You only need to modify the species name when running. Modifications are required in train_label and
   test_label, specifically the number of positive and negative samples in the training and test sets.
# ## Dependency

| Main Package  | Version |
|---------------|--------:|
| Python      	 |     3.7 |
| keras       	 |   2.6.0 |
| tensorflow    |   2.6.0 |
| Pandas      	 |   1.3.5 | 
| Scikit-learn  |  0.24.2 |
| biopython     |    1.81 |
| tqdm          |  4.66.2 |
| fair-esm      |   2.0.0 |
| torch         |   1.7.1 |
| matplotlib    |   3.5.3 |
| numpy         |  1.21.6 |
