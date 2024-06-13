# BCG-code
Code repository of the paper titled "Better Call Graphs: A New Dataset of Function Call Graphs for Malware Classification"


## Directory structure ##
### **LDP**: contains source code to reproduce the results of LDP and APK features 
* **main.py**: program to produce the LDP and APK feature results.

### **GNN**: contains source code to reproduce the results of GNN methods (GCN, GIN, and GraphSAGE). Note that, we utilize the code from Malnet paper and adapted to our experimental setup.

### datasets: It should contain the dataset. Please download the FCG dataset and APk features files from this link: https://jakir-sust.github.io/BCG-dataset/. 


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
    
* Run the code: first goto the code directory. Then run:
    
       python main.py  --exp_type $TYPE --data_type $DATA --rem_dup $REM_DUP --group $GROUP
        e.g., python main.py  --exp_type APK --data_type BCG --rem_dup 1 --group type
* Arguments are optional. Options are,
  * **exp_type**: defines the experiment type.
  * **data_type**: defines the data type. Options are: 1) BCG, 2) tiny, 3) Maldroid


### Results
 * Executing the code will create a new file contaning the detail results inside results folder for the given experiment and dataset type.



<!-- ## Experimental Datasets
 Need to be updated. -->

<!-- ## Note
This code was obtained by request from the corresponding authors. -->