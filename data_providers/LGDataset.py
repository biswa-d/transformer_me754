"""torch dataset for the LG Dataset"""
import torch

from torch.utils.data import Dataset, DataLoader

import pandas as pd

from sklearn.preprocessing import StandardScaler

class SequentialLGDataset(Dataset):
    def __init__(self,
                 path:str,
                 sequence_length:int,
                 sampling:int=1,
                )->None:
        super(SequentialLGDataset, self).__init__()
        self.sequence_length = sequence_length
        

        # read the data
        data = pd.read_csv(path)[::sampling].values
        self.data = data

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # for the encoder
        start_idx = idx
        end_idx = idx + self.sequence_length

        sequence = torch.tensor(self.data[start_idx:end_idx,:-1], dtype=torch.float32)

        voltage = sequence[:, 0]
        current = sequence[:, 1]
        temp = sequence[:, 2]

        volt_temp_cat = torch.cat((voltage.reshape(-1,1), temp.reshape(-1,1)), dim=1) 
        current_temp_cat = torch.cat((current.reshape(-1,1), temp.reshape(-1,1)), dim=1)

        # for the decoder and target
        target_start_idx = start_idx + 1
        target_end_idx = target_start_idx + self.sequence_length

        target_sequence = torch.tensor(self.data[target_start_idx:target_end_idx,:-1], dtype=torch.float32)
        volt = target_sequence[:, 0]
        current = target_sequence[:, 1]
        temp = target_sequence[:, 2]

        target_current_temp_cat = torch.cat((current.reshape(-1,1), temp.reshape(-1,1)), dim=1)
        target_volt_temp_cat = torch.cat((volt.reshape(-1,1), temp.reshape(-1,1)), dim=1)

        # output
        soc = torch.tensor(self.data[target_start_idx:target_end_idx, -1], dtype=torch.float32)

        return volt_temp_cat, current_temp_cat,target_volt_temp_cat, target_current_temp_cat, soc.unsqueeze(1)
    

def get_dataloader(dataset:Dataset, batch_size:int, shuffle=True)->DataLoader:
    '''returns the dataloader for the LG dataset
    
    Parameters
    ----------
    dataset : Dataset
        the dataset to be used
    batch_size : int
        the batch size
    shuffle : bool
        whether to shuffle the dataset

    Returns
    -------
    DataLoader
        dataloader for the LG dataset
    '''
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader