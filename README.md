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
  - Step 1: In this model, only the esm_main.py file in the esm folder and the first layer main.py file in the codes folder need to be run. The rest are auxiliary files.
  - Step 2:First, run the esm_main.py file:
a) In this py file, change the species name on line 12 to C. jejuni.
b) Change the code on lines 15-17 to train and run the program to obtain the ESM-encoded data file for the train dataset of this species. The second time, change the code to test to obtain the ESM-encoded data file for the test dataset of this species. No other operations are required for this file.
  - Step 3: Next, run the first layer main.py file and modify the species name on line 13. Running this file will produce the results shown in the paper.

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
