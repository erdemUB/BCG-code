import torch
import numpy as np
from glob import glob

from torch_geometric.data import Dataset


class MalnetDataset(Dataset):
    """
    Custom Dataset class for loading graph data from .pt files.
    """

    def __init__(self, args, root, files, labels, transform=None, pre_transform=None, sha_dict=None):
        """
        args: command line arguments
        root: root directory where the dataset is stored
        files: list of .pt files containing graph data
        labels: list of labels corresponding to each graph
        sha_dict: dictionary mapping indices to SHA identifiers for the files
        transform: optional transform to be applied on a sample
        """

        self.args = args
        self.files = files
        self.labels = labels
        self.num_labels = len(np.unique(self.labels))
        self.sha_dict = sha_dict

        super(MalnetDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.files

    @property
    def processed_file_names(self):
        return glob(self.processed_dir.replace('/processed', '') + '/*.pt')

    def download(self):
        pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sha_id = self.sha_dict[idx]
        x = torch.load(self.processed_dir.replace('/processed', '') + '/data_{}.pt'.format(sha_id))
        x.y = self.labels[idx]

        return x