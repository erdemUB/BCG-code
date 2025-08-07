# BCG-code
Code repository of the paper titled "Better Call Graphs: A New Dataset of Function Call Graphs for Malware Classification"


## Directory structure ##
### **LDP**: 
Contains source code to reproduce the results of LDP and APK features 
* **main.py**: program to produce the LDP and APK feature results.
* **dataset.py**: contains the function to create in memory dataset.
* **apk_feature**: APK features are split into train, test, and validation sets for the BCG dataset.
For other datasets (e.g., MalDroid), the APK features should be split similarly and placed
in the corresponding folders.

[//]: # (* **processed_data**: contains in memory dataset.)


### **GNN**: 
This repository contains source code to reproduce the results of Graph Neural Network (GNN) methods, including **GCN**, **GIN**, and **GraphSAGE**.
Our implementation builds upon the official codebase of the [MalNet paper](https://github.com/safreita1/malnet-graph), which has been adapted to fit our experimental setup.
### datasets

It should contain the **hashed FCG dataset** and **APK features**.

Please download both the hashed FCG dataset and APK feature file from this link:  
[https://erdemub.github.io/BCG-dataset/](https://erdemub.github.io/BCG-dataset/#Download)

- Place all **hashed FCG** files inside the `datasets/BCG_hashed_FCGs/` directory.  
- Place the **APK features (BCGAllAPKFeatures.json)** file inside the `datasets/` directory.

### results

Executing the code will generate detailed results inside the `results` folder.  
The results are organized by experiment type (e.g., `type` or `family`), dataset (`BCG`, `MalDroid`, or `MalNet Tiny`), and various hyperparameter configurations.

<!-- ## Requirements -->
<!-- ```bash -->
<!-- * C++ -->

## How to Run the Code? ##



### Required Libraries

Before running the code, install the following dependencies:

```bash
sudo apt install libcairo2-dev
pip install torch
pip install torch-geometric
pip install networkx==2.6.3
pip install joblib
pip install tqdm
```


### Execution Instructions
To run the code, follow these steps:
    
  * Run the LDP code: Navigate to the LDP directory and execute:
  
         python main.py  --exp_type $TYPE --data_type $DATA --rem_dup $REM_DUP --group $GROUP --graph_path $GRAPH_PATH | tee ../results/output.log
         e.g., python main.py  --exp_type APK --data_type BCG --rem_dup 1 --group type --graph_path ../datasets/BCG_hashed_FCGs/ | tee ../results/output.log
    Arguments are optional. Options are,
    * **exp_type**: defines the experiment type. Options are: APK, LDP, LDP+APK, APK_only, and Graph_only.
    * **data_type**: defines the data type. Options are: 1) BCG, 2) tiny, and 3) Maldroid
    * **rem_dup**: applicable for tiny and Maldroid. For BCG, default is 1.
    * **group**: applicable for BCG only. Options are type and family. For tiny and BCG default value is type.
    * **graph_path**: path of downloaded hashed graph files.

  * Run the GNN code: Navigate to the GNN code directory and execute:
  
        python gnn_experiments.py  --model $MODEL --data_type $data_type --rem_dup $REM_DUP --seed $SEED --group $GROUP
        e.g., python gnn_experiments.py  --model gcn --data_type BCG --rem_dup 1 --seed 0 --group type
    Arguments are optional. Options are similar to LDP above.

<!-- ## Experimental Datasets
 Need to be updated. -->

<!-- ## Note
This code was obtained by request from the corresponding authors. -->
