import torch as T
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


class ElementwiseScaler():
    # scales and unscales a vector elementwise
    # min and max should be np.arrays of the same shape as the vector x
    def __init__(self, min_unscaled, max_unscaled, min_scaled, max_scaled):
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
        shuffle=True, drop_last=True)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=len(val_dataset),
        shuffle=True)

    return train_dataloader, val_dataloader

def get_test_dataset(settings):
    dataset = load_raw_data(settings, train_or_test='test')
    dataset = scale_data(dataset, settings)
    
    # If 'dataset' is a pandas DataFrame, convert to tensor or extract columns accordingly
    if settings['process'] == 'CSTR1':
        print(dataset)

        # Extract X (Tr, Tj) and U (Fc)
        X = T.tensor(dataset[['Tr', 'Tj']].values, dtype=settings['accuracy'])  # State variables (Tr and Tj)
        U = T.tensor(dataset['Fc'].values, dtype=settings['accuracy']).unsqueeze(1)  # Control input (Fc)
        
        # Return X and U as a dictionary
        processed_dataset = {
            'X': X,
            'U': U
        }
    else:
        raise ValueError(f"Process {settings['process']} not implemented.")
    
    return processed_dataset

def get_test_dataloader1(settings):
    dataset = load_raw_data(settings, train_or_test='test')
    dataset = scale_data(dataset, settings)
    X0_test, U0_test, X1_test = get_X0_U0_X1(dataset, settings)
    dataset = TensorDataset(
        T.tensor(X0_test, dtype=settings['accuracy']),
        T.tensor(U0_test, dtype=settings['accuracy']),
        T.tensor(X1_test, dtype=settings['accuracy']))
    test_dataloader = DataLoader(
        dataset=dataset,
        batch_size=len(dataset))
    return test_dataloader

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
        min_scaled = np.array([-1.0, -1.0, -1.0])
        max_scaled = np.array([1.0, 1.0, 1.0])
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
    #unsure
    # if random_split:
    #     np.random.shuffle(indices)
    split_index = int(np.floor(settings['train_val_ratio'] * len(dataset)))
    train_indices, val_indices = indices[:split_index], indices[split_index:]

    # Use .iloc for pandas DataFrame
    train_dataset = dataset.iloc[train_indices]
    val_dataset = dataset.iloc[val_indices]
    return train_dataset, val_dataset

def get_X0_U0_X1(dataset, settings):
    # Use .iloc for pandas DataFrame
    X0 = dataset.iloc[:-1, [2, 3]].values  # don't need second column Tc - only one control input needed
    U0 = dataset.iloc[:-1, [0]].values
    X1 = dataset.iloc[1:, [2, 3]].values  # start from 1 
    return X0, U0, X1


if __name__ == '__main__':
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
