import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from constants import FOLDER, S_COLUMNS, M_COLUMNS, G_COLUMNS, LABEL


class CustomDataset(Dataset):
    def __init__(self, raw_df, join_df):
        self.raw_df = raw_df
        self.join_df = join_df

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        row = self.raw_df.iloc[idx]
        district = row['District']
        latitude = row['Latitude']
        longitude = row['Longitude']
        date_of_harvest = row['Date of Harvest']
        label = row[LABEL]

        inputs = self.join_df[(self.join_df['District'] == district) &
                              (self.join_df['Latitude'] == latitude) &
                              (self.join_df['Longitude'] == longitude) &
                              (self.join_df['Date of Harvest'] == date_of_harvest)]

        inputs['date'] = pd.to_datetime(inputs['date'], format='%d-%m-%Y')
        inputs = inputs.sort_values('date').reset_index(drop=True)
        s_inputs = torch.tensor(inputs[S_COLUMNS].values, dtype=torch.float)
        m_inputs = torch.tensor(inputs[M_COLUMNS].values, dtype=torch.float)
        g_inputs = torch.tensor(row[G_COLUMNS].astype('float64').values, dtype=torch.float)

        item = {
            'district': district, 
            'latitude': latitude, 
            'longitude': longitude, 
            'date_of_harvest': date_of_harvest,
            's_inputs': s_inputs,
            'm_inputs': m_inputs,
            'g_inputs': g_inputs,
            'label': label
        }  
        
        return item
    

def get_loaders(batch_size, num_workers):
    raw_train_df = pd.read_csv(f'../data/processed/lstm/{FOLDER}/raw_train.csv')
    join_train_df = pd.read_csv(f'../data/processed/lstm/{FOLDER}/join_train.csv')
    train_dataset = CustomDataset(raw_train_df, join_train_df)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    
    raw_test_df = pd.read_csv(f'../data/processed/lstm/{FOLDER}/raw_test.csv')
    join_test_df = pd.read_csv(f'../data/processed/lstm/{FOLDER}/join_test.csv')
    test_dataset = CustomDataset(raw_test_df, join_test_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    return train_loader, test_loader


class get_lstm_output(nn.Module):
    def forward(self, x):
        return x[0]
    

def lstm(sequence_length, num_features, hidden_size, num_layers, dropout=0.1):
    model = nn.Sequential(
        nn.LSTM(num_features, hidden_size, num_layers),
        get_lstm_output(),
        nn.BatchNorm1d(sequence_length),
        nn.Tanh(),
        nn.Dropout(dropout))
    return model


def conv1d(in_channels, out_channels, kernel_size=3, dropout=0.1):
    model = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Dropout(dropout))
    return model


def fc(in_features, dropout=0.1):
    model = nn.Sequential(
        nn.Linear(in_features, 4*in_features),
        nn.ReLU(),
        nn.Linear(4*in_features, 2*in_features),
        nn.BatchNorm1d(2*in_features),
        nn.ReLU(),
        nn.Dropout(dropout))
    return model


def concat_inputs(s_output, m_output, g_output):
    flatten = nn.Flatten()
    s_output = flatten(s_output)
    m_output = flatten(m_output)
    f_output = torch.cat((s_output, m_output, g_output), 1)
    return f_output


def last_fc(in_features):
    model = nn.Sequential(
        nn.Linear(in_features, int(in_features/2)),
        nn.ReLU(),
        nn.Linear(int(in_features/2), int(in_features/4)),
        nn.ReLU(),
        nn.Linear(int(in_features/4), 1))
    return model


class CustomModel(nn.Module):
    def __init__(self, sequence_length, num_layers, hidden_size, s_num_features, m_num_features, g_in_features, c_in_features):
        super().__init__()
        self.s_lstm = lstm(sequence_length, s_num_features, hidden_size, num_layers)
        self.s_conv1d = conv1d(sequence_length, hidden_size)

        self.m_lstm = lstm(sequence_length, m_num_features, hidden_size, num_layers)
        self.m_conv1d = conv1d(sequence_length, hidden_size)

        self.g_fc = fc(g_in_features)
        self.c_fc = last_fc(c_in_features)

    def forward(self, x):
        s_inputs = x['s_inputs']
        m_inputs = x['m_inputs']
        g_inputs = x['g_inputs']
        
        # Spectral inputs (LSTM + Conv1D)
        s_output = self.s_lstm(s_inputs)
        s_output = self.s_conv1d(s_output)

        # Meteo inputs (LSTM + Conv1D)
        m_output = self.m_lstm(m_inputs)
        m_output = self.m_conv1d(m_output)
        
        # Geo inputs (Fully connected layers)
        g_output = self.g_fc(g_inputs)

        # Concatanate inputs
        c_output = concat_inputs(s_output, m_output, g_output)
        
        # print(c_output.shape) to get c_in_features
        
        # Concat inputs (Fully connected layers)
        output = self.c_fc(c_output)

        return output
    
    
if __name__ == "__main__":
    train_loader, test_loader = get_loaders(batch_size=4, num_workers=4)
    first_batch = next(iter(train_loader))

    hidden_size = 32 # try 32, 64, 128
    num_layers = 2 # try 1, 2, 3, 4
    sequence_length = first_batch['s_inputs'].shape[1]
    s_num_features = first_batch['s_inputs'].shape[2]
    m_num_features = first_batch['m_inputs'].shape[2]
    g_in_features = first_batch['g_inputs'].shape[1]
    c_in_features = 1926

    model = CustomModel(sequence_length, num_layers, hidden_size, s_num_features, m_num_features, g_in_features, c_in_features)
    print(model(first_batch))