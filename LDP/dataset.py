import torch
# from torch_geometric.datasets import MalNetTiny
import torch_geometric.transforms as T
import pandas as pd

from tqdm import tqdm



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
    """
    A custom dataset class for loading and processing graph data from edge list files.
    """

    def __init__(
            self,
            args: None,
            root: str,
            data_frame: pd.DataFrame,
            split=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        """
        Initialize the dataset with given parameters and load processed data if available.

        args: command-line arguments
        root: root directory for the dataset
        data_frame: pandas DataFrame containing metadata for each graph
        transform: optional transformation to apply to each Data object
        """

        self.data_frame = data_frame
        self.args = args
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
        """
        Process the raw data files to create a list of Data objects and save them.

        1. Reads each edge list file corresponding to the graphs.
        2. Constructs edge_index tensors and Data objects.
        3. Saves the processed data to disk.
        """
        data_list = []

        # Set the hashed graph path as base path to read each graph
        y_map = {'adware': 0, 'benign': 1, 'downloader': 2, 'trojan': 3, 'addisplay': 4}

        for index, row in self.data_frame.iterrows():
            path = osp.join(self.args.graph_path, f"{row['sha256']}.edgelist")
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
    """
    Dataset class for the Maldroid dataset, inheriting from InMemoryDataset.
    """

    def __init__(
            self,
            root: str,
            data_frame: pd.DataFrame,
            split=None,
            args=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.data_frame = data_frame
        self.args = args
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
        """
        Process the raw data files to create a list of Data objects and save them.

        1. Reads each edge list file corresponding to the graphs.
        2. Constructs edge_index tensors and Data objects.
        3. Saves the processed data to disk.
        """
        
        data_list = []
        y_map = {'Benign': 0, 'Riskware': 1, 'Banking': 2, 'Adware': 3}
        for index, row in self.data_frame.iterrows():
            path = osp.join(self.args.graph_path, f"{row['sha256']}.edgelist")
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
    """A custom dataset class for loading and processing graph data from edge list files."""

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
        """
        Initialize the dataset with given parameters and load processed data if available.

        args: command-line arguments
        label_values: list of possible label values for classification
        root: root directory for the dataset
        data_frame: pandas DataFrame containing metadata for each graph
        transform: optional transformation to apply to each Data object
        """

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
        """
        Process the raw data files to create a list of Data objects and save them.

        1. Reads each edge list file corresponding to the graphs.
        2. Constructs edge_index tensors and Data objects.
        3. Saves the processed data to disk.
        """

        print("Inside process, No processed data exist")
        data_list = []
        y_map = {}
        if self.args.group == 'family':
            y_map.clear()
            print("Inside family   ")
            for i, value in enumerate(self.label_values):
                y_map[value] = i
        elif self.args.group == 'type':
            y_map.clear()
            print("Inside type   ")
            for i, value in enumerate(self.label_values):
                y_map[value] = i

        print("all class type and length: ", len(y_map), y_map)

        for index, row in tqdm(self.data_frame.iterrows(), total=len(self.data_frame)):
            # path = osp.join(base_path, 'graph_files1', f"{row['sha256']}.edgelist")
            path = osp.join(self.args.graph_path, f"{row['sha256']}.edgelist")
            malware_type = row['final_label']
            if self.args.group == 'family':
                malware_type = row['family_label']
            y = y_map.setdefault(malware_type, len(y_map))
            sha256 = row['sha256']

            with open(path, 'r') as f:
                edges = f.read().split('\n')[5:-1]

            edge_index = []
            for edge in edges:
                """Parse each edge line and convert to integer indices."""
                parts = edge.strip().split("\t")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    edge_index.append([int(parts[0]), int(parts[1])])

            # edge_index = [[int(s) for s in edge.split()] for edge in edges]
            edge_index = torch.tensor(edge_index).t().contiguous()
            num_nodes = int(edge_index.max()) + 1
            data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
            data.sha256 = sha256
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            """Apply pre-transformations to each Data object in the list."""
            data_list = [self.pre_transform(data) for data in data_list]

        # print(y_map)
        torch.save(self.collate(data_list), self.processed_paths[0])