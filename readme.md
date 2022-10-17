# HIGH-PPI
Hierarchical Graph Learning for Protein-Protein Interaction
## Dependencies
HIGH-PPI runs on Python 3.7-3.9. To install all dependencies, run:
```
pip install -r requirements.txt
```
# Datasets
Three datasets (SHS27k, SHS148k and STRING) can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1Yb-fdWJ5vTe0ePAGNfrUluzO9tz1lHIF?usp=sharing):
* `protein.actions.SHS27k.STRING.pro2.txt`             PPI network of SHS27k
* `protein.SHS27k.sequences.dictionary.pro3.tsv`      Protein sequences of SHS27k
* `protein.actions.SHS148k.STRING.txt`             PPI network of SHS148k
* `protein.SHS148k.sequences.dictionary.tsv`         Protein sequences of SHS148k
* `9606.protein.action.v11.0.txt`         PPI network of STRING
* `protein.STRING_all_connected.sequences.dictionary.tsv`             Protein sequences in STRING
* `edge_list_12`             Adjacency matrix for all proteins in SHS27k
* `x_list`             Feature matrix for all proteins in SHS27k

# PPI Prediction

Example: predicting unknown PPIs in SHS27k datasets with native structures:
First, generate adjacency matrix with native PDB files:
```
python generate_adj.py --distance 12
```
Then, generate feature matrix:
```
python generate_feat.py
```
To predict PPIs, use 'train_main.py' script with the following options:
* `ppi_path`             str, PPI network information
* `pseq_path`             str, Protein sequences
* `p_feat_matrix`       str, The feature matrix of all protein graphs
* `split`       str, Dataset split mode
* `save_path`             str, Path for saving models, configs and results
python train_main_sag_copy.py --ppi_path ./protein.actions.SHS27k.STRING.pro2.txt --pseq ./protein.SHS27k.sequences.dictionary.pro3.tsv --split random --p_feat_matrix ./x_list_7_new.pt --p_adj_matrix ./edge_list_12.npy --save_path ./result_save6 --epoch_num 500
```