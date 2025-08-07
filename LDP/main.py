# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

import requests
import json
import os
import pickle


# import seaborn as sns

from glob import glob
from tqdm import tqdm
import itertools
import gc

import numpy as np

from androguard.core.apk import APK
import networkx as nx
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
from androguard.misc import AnalyzeAPK
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_classif


from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, roc_auc_score, accuracy_score, precision_recall_fscore_support
from pprint import pprint

# pd.options.display.max_colwidth = 500

from dataset import *

import torch
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_std

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree
import argparse
import warnings

from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('always')

def parameter_parser():
    """Parse up command line parameters."""

    parser=argparse.ArgumentParser(description="LDP parser")
    parser.add_argument('--data_type', default="BCG", help='Type of data: BCG, tiny, Maldroid')
    parser.add_argument('--rem_dup', default=0, type=int, help='Duplicate remove or not')
    parser.add_argument('--exp_type', default="LDP", type=str, help='Type of experiments: LDP, APK, APK_only, LDP+APK')
    parser.add_argument('--seed', default=1, type=int, help='Seed values')
    parser.add_argument('--group', default="type", type=str, help='Classification group: type or family')
    parser.add_argument('--graph_path', default="../datasets/BCG_hashed_FCGs/", type=str, help='Path of APK graph files')
    parser.add_argument('--split_type', default="random", type=str, help='random or time_split or half1 or half2')

    return parser.parse_args()

@functional_transform('local_degree_profile3')
class LocalDegreeProfile(BaseTransform):
    """Computes the local degree profile (LDP) of each node in the graph"""

    def __call__(self, data: Data) -> Data:
        bins = 32
        row, col = data.edge_index
        N = data.num_nodes

        deg = degree(row, N, dtype=torch.float)
        deg_col = deg[col]

        min_deg, _ = scatter_min(deg_col, row, dim_size=N)
        min_deg[min_deg > 10000] = 0
        max_deg, _ = scatter_max(deg_col, row, dim_size=N)
        max_deg[max_deg < -10000] = 0
        mean_deg = scatter_mean(deg_col, row, dim_size=N)
        std_deg = scatter_std(deg_col, row, dim_size=N)

        features = torch.stack([deg, min_deg, max_deg, mean_deg, std_deg], dim=1)

        embedding = []
        for i in range(features.shape[1]):
            x = features[:, i]
            emb = torch.histogram(x, bins=bins, range=(0.0, 10.0))[0]
            embedding.append(emb)
        #         embedding = torch.concat(torch.Tensor(embedding), dim=-1)
        embedding = np.concatenate(embedding).reshape(-1)

        data.x = torch.Tensor(embedding)
        return data

def apply_lambda(df, data_type="BCG", rem_dup=True):
    """
    Apply lambda functions to specific columns in the DataFrame.

    df=DataFrame
    data_type=BCG or tiny or Maldroid
    rem_dup=remove duplicate or not
    """

    df['app_perm'] = df['app_perm'].apply(lambda x: ';'.join(x))
    df['app_all_act'] = df['app_all_act'].apply(lambda x: ';'.join(x))
    df['services'] = df['services'].apply(lambda x: ';'.join(x))
    df['receivers'] = df['receivers'].apply(lambda x: ';'.join(x))
    df['library'] = df['library'].apply(lambda x: ';'.join(x))

    df['app_main_act'] = df['app_main_act'].fillna('')
    df['app_main_act'] = df['app_main_act'].apply(lambda x: x.split('.')[-1])

    # assuming the column name is 'column_name'
    df['app_all_act'] = df['app_all_act'].apply(lambda x: ';'.join([s.split('.')[-1] for s in x.split(';')[1:]]))
    if data_type == 'BCG':
        df['family_label'] = df['family_label'].replace('', 'no_family')
        df['family_label'] = df['family_label'].fillna('no_family')
    return df


# import numpy as np
def load_apk_feature(train_dataset, test_dataset, val_dataset, data_type='tiny', exp_type="LDP+APK"):
    """
    Load APK features and combine them with graph embeddings based on the experiment type.

    exp_type: LDP, APK, APK_only, LDP+APK, LDP+APK_only, LDP+Graph_only, Graph_only
    data_type: BCG, tiny, Maldroid
    train_dataset, test_dataset, val_dataset: PyG datasets
    return: train_embed, train_label, val_embed, val_label, test_embed, test_label
    """

    global train_feat, test_feat, val_feat
    cols = ['apk_size', 'dex_size', 'apk_size_mb', 'dex_size_mb', 'app_name_pca_1', 'app_name_pca_2', 'package_name_pca_1', 'package_name_pca_2',
            'app_perm_pca_1', 'app_perm_pca_2', 'app_main_act_pca_1', 'app_main_act_pca_2', 'app_all_act_pca_1', 'app_all_act_pca_2', 'services_pca_1',
            'services_pca_2', 'receivers_pca_1', 'receivers_pca_2', 'library_pca_1', 'library_pca_2', 'nodes', 'edges', 'node_degree',
            'indegree', 'large_conn_dic', 'large_weak_conn_dic']  # , 'selfloop', 'closeness', 'num_cycle', 'large_conn_ratio_dic',
    # 'large_weak_conn_ratio_dic', 'second_large_weak_conn_dic', 'second_large_weak_conn_ratio_dic', 'power_alpha', 'power_sigma']
    if "APK_only" in exp_type:
        cols = ['apk_size', 'dex_size', 'apk_size_mb', 'dex_size_mb', 'app_name_pca_1', 'app_name_pca_2', 'package_name_pca_1', 'package_name_pca_2',
                'app_perm_pca_1', 'app_perm_pca_2', 'app_main_act_pca_1', 'app_main_act_pca_2', 'app_all_act_pca_1', 'app_all_act_pca_2', 'services_pca_1',
                'services_pca_2', 'receivers_pca_1', 'receivers_pca_2', 'library_pca_1', 'library_pca_2']
        print("APK only  ", train_feat.columns)
        # cols = ['app_perm', 'app_all_act', 'services', 'receivers', 'library', 'app_main_act']
    elif "Graph_only" in exp_type:
        cols = ['edges', 'nodes', 'node_degree', 'indegree', 'large_conn_dic', 'large_weak_conn_dic']

    if args.data_type == 'BCG':
        def fill_with_random(col):
            non_null_values = col.dropna().values
            def replace_nan(x):
                try:
                    if isinstance(x, (list, np.ndarray)):
                        return x if not pd.isna(x).all() else np.random.choice(non_null_values)
                    return np.random.choice(non_null_values) if pd.isna(x) else x
                except Exception:
                    return x
            return col.apply(replace_nan)

        nan_counts = test_feat[['nodes', 'edges']].isna().sum()
        print("test nan counts:", nan_counts)

        train_feat = train_feat.apply(fill_with_random)
        val_feat = val_feat.apply(fill_with_random)
        test_feat = test_feat.apply(fill_with_random)

    if exp_type == "LDP+APK" or exp_type == "LDP+APK_only" or exp_type == "LDP+Graph_only":
        train_embed = [np.concatenate((data.x.numpy(), train_feat[train_feat['sha256'] == data.sha256][cols].values[0]), axis=0)
                       for _, data in enumerate(train_dataset)]

        val_embed = [np.concatenate((data.x.numpy(), val_feat[val_feat['sha256'] == data.sha256][cols].values[0]), axis=0)
                      for _, data in enumerate(val_dataset)]
        test_embed = [np.concatenate((data.x.numpy(), test_feat[test_feat['sha256'] == data.sha256][cols].values[0]), axis=0)
                      for _, data in enumerate(test_dataset)]
    elif exp_type == "APK" or exp_type == "APK_only" or exp_type == "Graph_only":
        train_embed = [train_feat[train_feat['sha256'] == data.sha256][cols].values[0] for _, data in enumerate(train_dataset)]
        val_embed = [val_feat[val_feat['sha256'] == data.sha256][cols].values[0] for _, data in enumerate(val_dataset)]
        test_embed = [test_feat[test_feat['sha256'] == data.sha256][cols].values[0] for _, data in enumerate(test_dataset)]
    elif exp_type == "LDP":
        train_embed = [data.x.numpy() for _, data in enumerate(train_dataset)]
        val_embed = [data.x.numpy() for _, data in enumerate(val_dataset)]
        test_embed = [data.x.numpy() for _, data in enumerate(test_dataset)]

    train_label = [data.y.numpy()[0] for _, data in enumerate(train_dataset)]
    val_label = [data.y.numpy()[0] for _, data in enumerate(val_dataset)]
    test_label = [data.y.numpy()[0] for _, data in enumerate(test_dataset)]

    train_embed = np.array(train_embed)
    val_embed = np.array(val_embed)
    test_embed = np.array(test_embed)

    return train_embed, train_label, val_embed, val_label, test_embed, test_label


def grid_search(args, x_train, y_train, x_val, y_val):
    """
    Perform grid search to find the best hyperparameters for RandomForestClassifier.

    args: command line arguments
    x_train, y_train: training data and labels
    x_val, y_val: validation data and labels
    return: best_params, best_score
    """

    best_score = 0

    n_estimators = [1, 5, 10, 50]
    max_depths = [1, 5, 10, 20]
    params = list(itertools.product(n_estimators, max_depths))

    for n_estimator, max_depth in tqdm(params):
        clf = RandomForestClassifier(random_state=args.seed, n_estimators=n_estimator, max_depth=max_depth)
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_val)
        score = f1_score(y_val, y_pred, average='macro')

        if score > best_score:
            best_score = score
            best_params = {'n_estimators': n_estimator, 'max_depth': max_depth}

        del clf
        gc.collect()

    # print('\nBest val {}: {}'.format(args['metric'], best_score))

    return best_params, best_score

def perform_classifier(train_embed, train_label, val_embed, val_label, test_embed, test_label):
    """
    Train and evaluate the RandomForestClassifier.

    train_embed, train_label: training data and labels
    val_embed, val_label: validation data and labels
    test_embed, test_label: test data and labels
    return: accuracy, macro_f1, precision, recall
    """

    label1 = {'Benign': 0, 'Riskware': 1, 'Banking': 2, 'Adware': 3}
    label_dict = {v: k for k, v in label1.items()}
    class_indexes = list(label_dict.keys())
    class_labels = list(label_dict.values())

    # if args.data_type != "BCG":
    # clf = RandomForestClassifier(random_state=0)
    scalar = StandardScaler()
    train_embed = scalar.fit_transform(train_embed)
    test_embed, val_embed = scalar.transform(test_embed), scalar.transform(val_embed)

    params, val_score = grid_search(args, train_embed, train_label, val_embed, val_label)

    print("Grid search done ")

    clf = RandomForestClassifier(random_state=args.seed)
    clf.set_params(**params)
    clf.fit(train_embed, np.array(train_label))
    preds = clf.predict(test_embed)

    macro_f1 = round(f1_score(test_label, preds, average='macro'), 3)
    precision, recall, macroF1, _ = precision_recall_fscore_support(test_label, preds, average='macro')

    # precision = round(f1_score(test_label, preds, average='macro'), 3)
    # recall = round(f1_score(test_label, preds, average='macro'), 3)
    score = accuracy_score(test_label, preds)
    print(score, macro_f1)
    print(precision, recall, macroF1)

    report = classification_report(test_label, preds, labels=class_indexes, target_names=class_labels, zero_division=0)
    # report = classification_report(test_label, preds, labels=np.unique(preds), target_names=class_labels)
    return score, macro_f1, precision, recall

def create_Maldroid_dataset(transform, args, data=None):
    """
    Create the Maldroid dataset, applying transformations and handling duplicates.

    args: command line arguments
    transform: PyG transform to apply
    return: train_dataset, test_dataset, val_dataset
    """

    pr_path = "processed_data/Maldroid/"
    if args.rem_dup:
        pr_path = pr_path + 'unique/'
        emb_path = "temp_data/unique_Maldroid_train_test_val_dataset.pickle"
    else:
        pr_path = pr_path + 'duplicate/'
        emb_path = "temp_data/duplicate_Maldroid_train_test_val_dataset.pickle"

    if os.path.isfile(emb_path):
        with open(emb_path, 'rb') as handle:
            train_dataset, test_dataset, val_dataset = pickle.load(handle)
        return train_dataset, test_dataset, val_dataset

    df = pd.read_json('../datasets/Maldroid_original_APK_info.json', orient='records', lines=True)
    df = apply_lambda(df, data_type=args.data_type, rem_dup=args.rem_dup)
    cols_to_check = ['final_label', 'edges', 'nodes', 'node_degree', 'indegree', 'large_conn_dic', 'large_weak_conn_dic']
    if args.rem_dup == 1:
        df = df[~df.duplicated(subset=cols_to_check, keep='first')]
    print("dataset shape   ", df.shape)

    train_dataset = MaldroidDataset(args=args, root=pr_path, data_frame=df[df['sha256'].isin(train_feat.sha256)], transform=transform, split='train')
    test_dataset = MaldroidDataset(args=args, root=pr_path, data_frame=df[df['sha256'].isin(test_feat.sha256)], transform=transform, split='test')
    val_dataset = MaldroidDataset(args=args, root=pr_path, data_frame=df[df['sha256'].isin(val_feat.sha256)], transform=transform, split='val')

    with open(emb_path, 'wb') as handle:
        pickle.dump([train_dataset, test_dataset, val_dataset], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_dataset, test_dataset, val_dataset

def create_MalNetTiny_dataset(transform, args, data=None):
    """
    Create the MalNet Tiny dataset, applying transformations and handling duplicates.

    args: command line arguments
    transform: PyG transform to apply
    return: train_dataset, test_dataset, val_dataset
    """

    pr_path = "processed_data/Malnet_Tiny/"
    if args.rem_dup:
        pr_path = pr_path + 'unique/'
        emb_path = "temp_data/unique_tiny_train_test_val_dataset.pickle"
    else:
        pr_path = pr_path + 'duplicate/'
        emb_path = "temp_data/duplicate_tiny_train_test_val_dataset.pickle"

    if os.path.isfile(emb_path):
        with open(emb_path, 'rb') as handle:
            train_dataset, test_dataset, val_dataset = pickle.load(handle)
        return train_dataset, test_dataset, val_dataset


    df = pd.read_json('../datasets/tiny_original_APK_info.json', orient='records', lines=True)
    cols_to_check = ['final_label', 'edges', 'nodes', 'node_degree', 'indegree', 'large_conn_dic', 'large_weak_conn_dic']
    if args.rem_dup == 1:
        df = df[~df.duplicated(subset=cols_to_check, keep='first')]
    print("dataset shape   ", df.shape)

    df = apply_lambda(df, data_type=args.data_type, rem_dup=args.rem_dup)
    print("dataset shape   ", df.shape)

    train_dataset = MalNetTiny(args=args, root=pr_path, data_frame=df[df['sha256'].isin(train_feat.sha256)], transform=transform, split='train')
    test_dataset = MalNetTiny(args=args,root=pr_path, data_frame=df[df['sha256'].isin(test_feat.sha256)], transform=transform, split='test')
    val_dataset = MalNetTiny(args=args, root=pr_path, data_frame=df[df['sha256'].isin(val_feat.sha256)], transform=transform, split='val')

    with open(emb_path, 'wb') as handle:
        pickle.dump([train_dataset, test_dataset, val_dataset], handle, protocol=pickle.HIGHEST_PROTOCOL)
    return train_dataset, test_dataset, val_dataset

def create_BCG_dataset(transform, args, data=None):
    """
    Create the BCG dataset. Load from pickle if exists, else create from raw data.

    args: command line arguments
    transform: PyG transform to apply
    return: train_dataset, test_dataset, val_dataset
    """

    pr_path = "processed_data/BCG/"
    if args.rem_dup:
        pr_path = pr_path + 'unique/'
        emb_path = "temp_data/unique_BCG_train_test_val_dataset.pickle"
        if args.group == 'family':
            pr_path = pr_path + 'family/'
            emb_path = "temp_data/unique_BCG_train_test_val_dataset_family.pickle"
        if args.split_type != 'random':
            pr_path = pr_path + args.split_type + "/"
            emb_path = "temp_data/BCG_train_test_val_dataset_" + args.split_type + ".pickle"
    else:
        pr_path = pr_path + 'duplicate/'
        emb_path = "temp_data/duplicate_BCG_train_test_val_dataset.pickle"

    if os.path.isfile(emb_path):
        print("Processed data exist and returning from: ", emb_path)
        with open(emb_path, 'rb') as handle:
            train_dataset, test_dataset, val_dataset = pickle.load(handle)
        return train_dataset, test_dataset, val_dataset

    print("dataset not exist, creating from all graphs")
    print("pr and emb path: ", pr_path, emb_path)

    df = pd.read_json('../datasets/BCGAllAPKFeatures.json', orient='records', lines=True)

    cols_to_check = ['final_label', 'edges', 'nodes', 'node_degree', 'indegree', 'large_conn_dic', 'large_weak_conn_dic']
    # if args.rem_dup == 1:
    #     df = df[~df.duplicated(subset=cols_to_check, keep='first')]

    df = apply_lambda(df, data_type=args.data_type, rem_dup=args.rem_dup)
    if args.group == 'family':
        label_values = df['family_label'].unique()
    elif args.group == 'type':
        label_values = df['final_label'].unique()
    print("BCG dataset shape   ", df.shape)

    train_dataset = BCG(args, label_values, root=pr_path, data_frame=df[df['sha256'].isin(train_feat.sha256)], transform=transform, split='train')
    test_dataset = BCG(args, label_values, root=pr_path, data_frame=df[df['sha256'].isin(test_feat.sha256)], transform=transform, split='test')
    val_dataset = BCG(args,label_values,  root=pr_path, data_frame=df[df['sha256'].isin(val_feat.sha256)], transform=transform, split='val')


    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    with open(emb_path, 'wb') as handle:
        pickle.dump([train_dataset, test_dataset, val_dataset], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_dataset, test_dataset, val_dataset


if __name__ == '__main__':
    """
    Main function to execute the workflow. Sets up device, loads datasets, extracts features, and performs classification.
    
    1. Set device (GPU if available, else CPU).
    2. Parse command line arguments.
    3. Load APK features from JSON files.
    4. Create datasets based on the specified data type (Maldroid, MalNet Tiny, BCG).
    5. Extract embeddings and labels based on the experiment type (LDP, APK, etc.).
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([T.ToDevice(device), LocalDegreeProfile()])

    # val_dataset = MalNetTiny(root=pr_path, transform=transform, split='val')
    args = parameter_parser()
    data_type = args.data_type

    print("device, data type and rem duplicate  ========= ", device, data_type, args.rem_dup)
    # exit()

    if args.rem_dup:
        train_feat_file = "train_data_apk_feature_unique.json"
        val_feat_file = "val_data_apk_feature_unique.json"
        test_feat_file = "test_data_apk_feature_unique.json"
    else:
        train_feat_file = "train_data_apk_feature.json"
        val_feat_file = "val_data_apk_feature.json"
        test_feat_file = "test_data_apk_feature.json"

    if args.data_type == "tiny":
        root_dir = "apk_feature/Malnet_Tiny/"
    elif args.data_type == "Maldroid":
        root_dir = "apk_feature/maldroid/"
    elif args.data_type == "BCG":
        root_dir = "apk_feature/BCG/"
        if args.group == 'family':
            root_dir = "apk_feature_family/BCG/"
        if args.split_type != 'random':
            root_dir = "apk_feature_time_split/" + args.split_type + "/"

    print("Apk feature path and train file: ", root_dir, root_dir + train_feat_file)

    train_feat = pd.read_json(root_dir + train_feat_file, lines=True)
    val_feat = pd.read_json(root_dir + val_feat_file, lines=True)
    test_feat = pd.read_json(root_dir + test_feat_file, lines=True)

    print("apk feature load done  ", train_feat.shape, val_feat.shape, test_feat.shape, train_feat[:2])
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")


    #For maldroid
    if data_type == 'Maldroid':
        train_dataset, test_dataset, val_dataset = create_Maldroid_dataset(transform, args)
    elif data_type == 'tiny':
        train_dataset, test_dataset, val_dataset = create_MalNetTiny_dataset(transform, args, data=None)
    elif data_type == 'BCG':
        train_dataset, test_dataset, val_dataset = create_BCG_dataset(transform, args)

    print("Dataset creation done  ", data_type, train_dataset, test_dataset)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")

    # exp_type = 'LDP+APK'
    # exp_type = 'LDP'
    # exp_type = 'APK'
    exp_type = args.exp_type
    train_embed, train_label, val_embed, val_label, test_embed, test_label = load_apk_feature(train_dataset, test_dataset, val_dataset, data_type=data_type, exp_type=exp_type)

    print("ApK feature load done and exp type ==  ", exp_type)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")

    acc_score, macro_f1_list, Pre, Rec = [], [], [], []
    for exp in range(0, 10):
        args.seed = exp
        score, macro_f1, precision, recall = perform_classifier(train_embed, train_label, val_embed, val_label, test_embed, test_label)
        acc_score.append(np.round(score * 100, 2))
        macro_f1_list.append(np.round(macro_f1 * 100, 2))
        Pre.append(np.round(precision * 100, 2))
        Rec.append(np.round(recall * 100, 2))
    print(np.round(np.mean(acc_score), 2), "±", np.round(np.std(acc_score), 2), end=',')
    print(np.round(np.mean(macro_f1_list), 2), "±", np.round(np.std(macro_f1_list), 2), end=',')
    print(np.round(np.mean(Pre), 2), "±", np.round(np.std(Pre), 2), end=',')
    print(np.round(np.mean(Rec), 2), "±", np.round(np.std(Rec), 2), end='\n')

