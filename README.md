# BCG-code
Code repository of the paper titled "Better Call Graphs: A New Dataset of Function Call Graphs for Malware Classification"


## Directory structure ##
### **LDP**: 
Contains source code to reproduce the results of LDP and APK features 
* **main.py**: program to produce the LDP and APK feature results.
* **dataset.py**: contains the function to create in memory dataset.
* **apk_feature**: contains both original APK information and graph information of each APK. APK feature information are splited into train, test, and validation for both unique and duplicate datasets. Please download the original BCG apk information json file from here: https://iclr.me/download_page.html.
* **processed_data**: contains in memory dataset.


### **GNN**: 
Contains source code to reproduce the results of GNN methods (GCN, GIN, and GraphSAGE). Note that, we utilize the code from Malnet paper and adapted to our experimental setup.

### datasets: 
It should contain the dataset. Please download the FCG dataset and APk features files from this link: https://iclr.me/. 

### results:
Executing the code will create a new file contaning the detail results inside results folder for the given experiment and dataset type.

<!-- ## Requirements -->
<!-- ```bash -->
<!-- * C++ -->

## How to Run the Code? ##



### Required libraries:
1. Install the following packages
   1. sudo apt install libcairo2-dev
   2. pip install pytorch-geometric
   3. pip install networkx==2.6.3
   4. pip install joblib
   5. pip install tqdm


### Execute
    
* Run the code: first go to the code directory of LDP or GNN. Then run the code of both approaches using the following guidelines,
  * Run the LDP code:
  
         python main.py  --exp_type $TYPE --data_type $DATA --rem_dup $REM_DUP --group $GROUP --graph_path $GRAPH_PATH
         e.g., python main.py  --exp_type APK --data_type BCG --rem_dup 1 --group type --graph_path GraphFiles/
    Arguments are optional. Options are,
    * **exp_type**: defines the experiment type. Options are: APK, LDP, LDP+APK, APK_only, and Graph_only.
    * **data_type**: defines the data type. Options are: 1) BCG, 2) tiny, and 3) Maldroid
    * **rem_dup**: applicable for tiny and Maldroid. For BCG, default is 1.
    * **group**: applicable for BCG only. Options are type and family. For tiny and BCG default value is type.
    * **graph_path**: path of the downloaded graph files.

  * Run the GNN code:
  
        python gnn_experiments.py  --model $MODEL --data_type $data_type --rem_dup $rem_dup --seed $SEED --group $GROUP


<!-- ## Experimental Datasets
 Need to be updated. -->

<!-- ## Note
This code was obtained by request from the corresponding authors. -->