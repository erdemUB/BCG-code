import torch
# from torch_geometric.datasets import MalNetTiny
import torch_geometric.transforms as T
import pandas as pd

import os.path as osp
from typing import Callable, List, Optional
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
    extract_zip,
)


class MalNetTiny(InMemoryDataset):

    def __init__(
            self,
            root: str,
            data_frame: pd.DataFrame,
            split=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.data_frame = data_frame
        self.y_map = {}
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['malnet-graphs-tiny', osp.join('split_info_tiny', 'type')]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'split_slices.pt']

    def process(self):
        data_list = []
        #         base_path = '/panasas/scratch/grp-erdem/malnet-graphs/malnetTiny/'
        base_path = '/projects/academic/erdem/gurvinder/scratch/Malnet_Tiny/'
        y_map = {'adware': 0, 'benign': 1, 'downloader': 2, 'trojan': 3, 'addisplay': 4}

        #         for filename in filenames:
        for index, row in self.data_frame.iterrows():
            path = osp.join(base_path, 'graph_files1', f"{row['sha256']}.edgelist")
            malware_type = row['final_label']
            y = y_map.setdefault(malware_type, len(y_map))
            sha256 = row['sha256']

            with open(path, 'r') as f:
                edges = f.read().split('\n')[5:-1]

            edge_index = [[int(s) for s in edge.split()] for edge in edges]
            edge_index = torch.tensor(edge_index).t().contiguous()
            num_nodes = int(edge_index.max()) + 1
            data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
            data.sha256 = sha256
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print(y_map)
        torch.save(self.collate(data_list), self.processed_paths[0])


class MaldroidDataset(InMemoryDataset):

    def __init__(
            self,
            root: str,
            data_frame: pd.DataFrame,
            split=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.data_frame = data_frame
        self.y_map = {}
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['malnet-graphs-tiny', osp.join('split_info_tiny', 'type')]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'split_slices.pt']

    def process(self):
        data_list = []
        #         base_path = '/panasas/scratch/grp-erdem/malnet-graphs/maldroid/'
        base_path = '/projects/academic/erdem/gurvinder/scratch/maldroid/'
        y_map = {'Benign': 0, 'Riskware': 1, 'Banking': 2, 'Adware': 3}

        #         for filename in filenames:
        for index, row in self.data_frame.iterrows():
            path = osp.join(base_path, 'graph_files1', f"{row['sha256']}.edgelist")
            malware_type = row['final_label']
            y = y_map.setdefault(malware_type, len(y_map))
            sha256 = row['sha256']

            with open(path, 'r') as f:
                edges = f.read().split('\n')[5:-1]

            edge_index = [[int(s) for s in edge.split()] for edge in edges]
            edge_index = torch.tensor(edge_index).t().contiguous()
            num_nodes = int(edge_index.max()) + 1
            data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
            data.sha256 = sha256
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print(y_map)
        torch.save(self.collate(data_list), self.processed_paths[0])


class BCG(InMemoryDataset):

    def __init__(
            self,
            args: None,
            label_values: None,
            root: str,
            data_frame: pd.DataFrame,
            split=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.data_frame = data_frame
        self.args = args
        self.label_values = label_values
        self.y_map = {}
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['malnet-graphs-tiny', osp.join('split_info_tiny', 'type')]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'split_slices.pt']

    def process(self):
        data_list = []
        base_path = '/panasas/scratch/grp-erdem/malnet-graphs/virusShare'
        base_path = '/panasas/scratch/grp-erdem/malnet-graphs/BCG/'
        y_map = {'trojan': 0, 'adware': 1, 'risktool': 2, 'adware++risktool': 3, 'smsreg': 4, 'riskware': 5, 'adware++trojan': 6,
                 'adware++riskware': 7, 'dropper++trojan': 8, 'spy++trojan': 9, 'rog': 10, 'risktool++spr': 11, 'risktool++trojan': 12,
                 'banker++trojan': 13, 'addisplay': 14, 'riskware++smsreg': 15, 'riskware++trojan': 16, 'fakeapp': 17,
                 'downloader++trojan': 18, 'risktool++riskware': 19, 'smsreg++trojan': 20, 'clicker++trojan': 21, 'fakeapp++trojan': 22,
                 'spy': 23, 'spr++trojan': 24, 'smsreg++spr': 25, 'backdoor': 26, 'backdoor++trojan': 27, 'benign': 28}

        #         for filename in filenames:
        print(len(y_map), y_map)
        if self.args.group == 'family':
            y_map.clear()
            # y_map = self.label_values
            print("Inside family   ")
            # self.data_frame['family_label'] = self.data_frame['family_label'].replace('', 'no_family')
            # self.data_frame['family_label'] = self.data_frame['family_label'].fillna('no_family')
            # Y_values = self.data_frame['family_label'].unique()
            for i, value in enumerate(self.label_values):
                y_map[value] = i

        print(len(y_map), y_map)

        for index, row in self.data_frame.iterrows():
            #             print(index, row)
            #             path = osp.join(base_path, row['folder'] ,'graph_files1' , f"{row['sha256']}.edgelist")
            path = osp.join(base_path, 'graph_files1', f"{row['sha256']}.edgelist")
            malware_type = row['final_label']
            if self.args.group == 'family':
                malware_type = row['family_label']
            y = y_map.setdefault(malware_type, len(y_map))
            sha256 = row['sha256']

            with open(path, 'r') as f:
                edges = f.read().split('\n')[5:-1]

            edge_index = [[int(s) for s in edge.split()] for edge in edges]
            edge_index = torch.tensor(edge_index).t().contiguous()
            num_nodes = int(edge_index.max()) + 1
            data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
            data.sha256 = sha256
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # print(y_map)
        torch.save(self.collate(data_list), self.processed_paths[0])