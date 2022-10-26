# HIGH-PPI
Hierarchical Graph Learning for Protein-Protein Interaction
## Dependencies
HIGH-PPI runs on Python 3.7-3.9. To install all dependencies, directly run:
```
cd HIGH-PPI-main
conda env create -f environment.yml
conda activate HIGH-PPI
```
# Datasets
Three datasets (SHS27k, SHS148k and STRING) can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1Yb-fdWJ5vTe0ePAGNfrUluzO9tz1lHIF?usp=sharing):
* `protein.actions.SHS27k.STRING.pro2.txt`             PPI network of SHS27k
* `protein.SHS27k.sequences.dictionary.pro3.tsv`      Protein sequences of SHS27k
* `protein.actions.SHS148k.STRING.txt`             PPI network of SHS148k
* `protein.SHS148k.sequences.dictionary.tsv`         Protein sequences of SHS148k
* `9606.protein.action.v11.0.txt`         PPI network of STRING
* `protein.STRING_all_connected.sequences.dictionary.tsv`             Protein sequences of STRING
* `edge_list_12`             Adjacency matrix for all proteins in SHS27k
* `x_list`             Feature matrix for all proteins in SHS27k

# PPI Prediction

Example: predicting unknown PPIs in SHS27k datasets with native structures:

## Using Processed Data for SHS27k Dataset

Download  `protein.actions.SHS27k.STRING.pro2.txt` and `protein.SHS27k.sequences.dictionary.pro3.tsv` to ./HIGH-PPI-main.

Download `edge_list_12` and `x_list` to ./HIGH-PPI-main/protein_info

## Data Processing for New Datasets
Prepare all related PDB files. Native protein structures can be downloaded in batches from the [RCSB PDB](https://www.rcsb.org/downloads), and predicted protein structures with errors can be downloaded from the [AlphaFold database](https://alphafold.ebi.ac.uk/). Put all of the PDB files in './protein_info/'

Generate adjacency matrix with native PDB files:
```
python ./protein_info/generate_adj.py --distance 12
```
Generate feature matrix:
```
python ./protein_info/generate_feat.py
```

## Training
To predict PPIs, use 'model_train.py' script to train HIGH-PPI with the following options:
* `ppi_path`             str, PPI network information
* `pseq_path`             str, Protein sequences
* `p_feat_matrix`       str, The feature matrix of all protein graphs
* `p_adj_matrix`       str, The adjacency matrix of all protein graphs
* `split`       str, Dataset split mode
* `save_path`             str, Path for saving models, configs and results
* 'epoch_num'     int, Training epochs
```
python model_train.py --ppi_path ./protein.actions.SHS27k.STRING.pro2.txt --pseq ./protein.SHS27k.sequences.dictionary.pro3.tsv --split random --p_feat_matrix ./protein_info/x_list.pt --p_adj_matrix ./protein_info/edge_list_12.npy --save_path ./result_save --epoch_num 500
```
## Testing
Run 'model_test.py' script to test HIGH-PPI with the following options:
* `ppi_path`             str, PPI network information
* `pseq_path`             str, Protein sequences
* `p_feat_matrix`       str, The feature matrix of all protein graphs
* `p_adj_matrix`       str, The adjacency matrix of all protein graphs
* `model_path`       str, Path for trained model
* `index_path`             str, Path for index being tested
```
python model_test.py --ppi_path ./protein.actions.SHS27k.STRING.pro2.txt --pseq ./protein.SHS27k.sequences.dictionary.pro3.tsv --p_feat_matrix ./x_list.pt --p_adj_matrix ./edge_list_12.npy --model_path ./result_save/gnn_training_seed_1/gnn_model_valid_best.ckpt --index_path ./train_val_split_data/train_val_split_1.json
```
## Output
The output after running 'model_test.py' includes:
* `valid_label_list` Real PPI labels for the test index
* `test_pre_result_list` Predicted PPI results for the test index
* `best_f1` Performance in terms of best-F1 score
* `aupr` Performance in terms of AUPR score
