import torch as T
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

class ElementwiseScaler():
    # scales and unscales a vector elementwise
    # min and max should be np.arrays of the same shape as the vector x
    def _init_(self, min_unscaled, max_unscaled, min_scaled, max_scaled):
        self.min_unscaled = min_unscaled
        self.max_unscaled = max_unscaled
        self.range_unscaled = max_unscaled - min_unscaled
        self.min_scaled = min_scaled
        self.max_scaled = max_scaled
        self.range_scaled = max_scaled - min_scaled

    def scale(self, x):
        x = (x - self.min_unscaled) / self.range_unscaled
        x = x * self.range_scaled + self.min_scaled
        return x

    def unscale(self, x):
        x = (x - self.min_scaled) / self.range_scaled
        x = x * self.range_unscaled + self.min_unscaled
        return x

def get_train_val_dataloaders(settings):
    dataset = load_raw_data(settings, train_or_test='train')
    dataset = scale_data(dataset, settings)
    train_dataset, val_dataset = train_val_split(dataset, settings)
    
    X0_train, U0_train, X1_train = get_X0_U0_X1(train_dataset, settings)
    X0_val, U0_val, X1_val = get_X0_U0_X1(val_dataset, settings)

    train_dataset = TensorDataset(
        T.tensor(X0_train, dtype=settings['accuracy']),
        T.tensor(U0_train, dtype=settings['accuracy']),
        T.tensor(X1_train, dtype=settings['accuracy']))
    val_dataset = TensorDataset(
        T.tensor(X0_val, dtype=settings['accuracy']),
        T.tensor(U0_val, dtype=settings['accuracy']),
        T.tensor(X1_val, dtype=settings['accuracy']))

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=settings['batch_size'],
        shuffle=False, drop_last=True)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=len(val_dataset))

    return train_dataloader, val_dataloader

def get_test_dataset(settings):
    dataset = load_raw_data(settings, train_or_test='test')
    dataset = scale_data(dataset, settings)
    for k, v in dataset.items():
        dataset[k] = T.tensor(v.values, dtype=settings['accuracy'])
    
    if settings['process'] == 'CSTR1':
        for k, v in dataset.items():
            dataset[k] = {
                'X': v[:,[1, 2]],  # Tr and Tj is state(columsn 1 and 2
                'U': v[:,0:1],     #only Fc
            }
    else:
        raise ValueError(f"Process {settings['process']} not implemented.")
    
    return dataset

def load_raw_data(settings, train_or_test):
    dataset = pd.read_csv(settings[f'{train_or_test}_data_path'])
    return dataset

def scale_data(dataset, settings):
    if settings['process'] == 'CSTR1':
        # Adjust the scaler initialization as needed
        min_unscaled = np.concatenate((
            settings['CSTR1']['state_scaling']['min_unscaled'],
            settings['CSTR1']['action_scaling']['min_unscaled']))
        max_unscaled = np.concatenate((
            settings['CSTR1']['state_scaling']['max_unscaled'],
            settings['CSTR1']['action_scaling']['max_unscaled']))
        min_scaled = np.array([-1.0, -1.0, -1.0, -1.0])
        max_scaled = np.array([1.0, 1.0, 1.0, 1.0])
        scaler = ElementwiseScaler(
            min_unscaled, max_unscaled,
            min_scaled, max_scaled)

        # scale data
        dataset = scaler.scale(dataset.values)
    else:
        raise ValueError(f"Process {settings['process']} not implemented.")
    return dataset

def train_val_split(dataset, settings, random_split=True):
    indices = list(range(len(dataset)))
    # if random_split:
    #     np.random.shuffle(indices)
    split_index = int(np.floor(settings['train_val_ratio'] * len(dataset)))
    train_indices, val_indices = indices[:split_index], indices[split_index:]

    # Use array indexing instead of iloc for NumPy arrays
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    return train_dataset, val_dataset

def get_X0_U0_X1(dataset, settings):
    X0 = dataset[:-1, [2, 3]]  # dont need last column
    U0 = dataset[:-1, [0]]  
    X1 = dataset[1:, [2, 3]]  # start from 1 

    return X0, U0, X1

if __name__ == '_main_':
    from main import get_settings
    settings = get_settings()

    # train_dataloader, val_dataloader = get_train_val_dataloaders(settings)

    # for X0_batch, U0_batch, X1_batch in train_dataloader:
    #     print(X0_batch.shape, U0_batch.shape, X1_batch.shape)
    #     print(X0_batch[0])
    #     print(X1_batch[0])
    #     print(U0_batch[0])
    #     break

    # for X0_val, U0_val, X1_val in val_dataloader:
    #     print(X0_val.shape, U0_val.shape, X1_val.shape)
    #     break

    # test_dataset = get_test_dataset(settings)

    # print('data loading done done')