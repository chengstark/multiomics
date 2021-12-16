from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np

class TablePandasDataset(Dataset):
    """Pandas dataset.    """
    def __init__(self, pd, cov_list, utility_tag='Target'):
        """
        Args:
            pd: Pandas dataframe,
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.Y_torch = torch.Tensor(np.vstack(pd[utility_tag].values).astype('float32')) 
        #stacking because they are saved as list of numpy arrays

        self.X_torch = torch.Tensor(pd[cov_list].to_numpy().astype('float32'))

    def __len__(self):
        return self.Y_torch.shape[0]

    def __getitem__(self, idx):
        # data = self.pd_torch[idx]
        Y = self.Y_torch[idx]
        X = self.X_torch[idx]
        output = [X,Y]
        return output
    
def get_dataloaders(data_pd,cov_list, utility_tag='Target',shuffle=True, num_workers = 8,
                          batch_size = 32,drop_last=False):
    dataloader = DataLoader(TablePandasDataset(data_pd, cov_list, utility_tag=utility_tag),batch_size=batch_size,shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    return dataloader

def import_data(rna_filename, motif_filename, peaks_filename):
    rna = pd.read_csv(rna_filename, index_col=[0]).rename({'Unnamed: 0': "Gene"}, axis=1).T
    motif = pd.read_csv(motif_filename, index_col=[0]).rename({'Unnamed: 0': "Gene"}, axis=1).T
    atac = pd.read_csv(peaks_filename, index_col=[0]).rename({'Unnamed: 0': 'Gene'}, axis=1).T
    multiomics = pd.concat([rna,motif,atac], axis=1)
    # multiomics = multiomics.sample(frac=1).reset_index(drop=True)
    multiomics["Target"] = rna.apply(lambda row: rna_proportion(row), axis=1).to_numpy()
    print(multiomics.index)

    idx_test = list(range(int(multiomics.shape[0] * .8) + 1, multiomics.shape[0]))
    set_tag = ['test' if i in idx_test else 'train' for i in range(len(multiomics))]
    multiomics["dataset"] = set_tag
    HIVcolumns = rna.columns[-6:]
    return multiomics, HIVcolumns.append(pd.Index(["Target"]))
    
def rna_proportion(row):
    hivRNA = row[-6:].to_numpy()
    RNA = row[:-6].to_numpy()
    return sum(hivRNA)/sum(RNA)