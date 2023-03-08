import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from constants import FOLDER, S_COLUMNS, M_COLUMNS, G_COLUMNS
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np 


class DLDataset(Dataset):
    def __init__(self, raw_df, join_df, test=False):
        self.raw_df = raw_df
        self.join_df = join_df
        self.test = test

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        row = self.raw_df.iloc[idx]
        district = row['District']
        latitude = row['Latitude']
        longitude = row['Longitude']
        date_of_harvest = row['Date of Harvest']
        
        if self.test:
            label = row['Predicted Rice Yield (kg/ha)']
        else:
            label = row['Rice Yield (kg/ha)']

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
            'labels': label
        }  
        
        return item
    

def get_loaders(batch_size, num_workers, val_size=0.2):
    raw_train_df = pd.read_csv(f'../data/processed/lstm/{FOLDER}/raw_train.csv')
    join_train_df = pd.read_csv(f'../data/processed/lstm/{FOLDER}/join_train.csv')
    dataset = DLDataset(raw_train_df, join_train_df)
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1 - val_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    
    raw_test_df = pd.read_csv(f'../data/processed/lstm/{FOLDER}/raw_test.csv')
    join_test_df = pd.read_csv(f'../data/processed/lstm/{FOLDER}/join_test.csv')
    test_dataset = DLDataset(raw_test_df, join_test_df, test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


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
        nn.Linear(in_features, in_features),
        nn.ReLU(),
        nn.Linear(in_features, in_features),
        nn.BatchNorm1d(in_features),
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
        nn.Linear(in_features, in_features),
        nn.ReLU(),
        nn.Linear(in_features, int(in_features/2)),
        nn.ReLU(),
        nn.Linear(int(in_features/2), int(in_features/4)),
        nn.ReLU(),
        nn.Linear(int(in_features/4), int(in_features/8)),
        nn.ReLU(),
        nn.Linear(int(in_features/8), 1))
    return model


class DLModel(nn.Module):
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
        
        # print(c_output.shape) # RuntimeError: mat1 and mat2 shapes cannot be multiplied
        
        # Concat inputs (Fully connected layers)
        output = self.c_fc(c_output)

        return output
    
    
def train_epoch(model, optimizer, criterion, train_loader):
    train_loss = 0
    
    for data in train_loader:
        keys_inputs = ['s_inputs', 'm_inputs', 'g_inputs']
        inputs = {key: data[key] for key in keys_inputs}
        labels = data['labels'].float()

        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        
        optimizer.step()
        
    train_loss /= len(train_loader.dataset)
    
    return train_loss


def val_epoch(model, val_loader):
    val_loss = 0
    val_labels = []
    val_preds = []
    
    for data in val_loader:
        keys_inputs = ['s_inputs', 'm_inputs', 'g_inputs']
        inputs = {key: data[key] for key in keys_inputs}
        labels = data['labels'].float()

        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        
        val_labels += labels.tolist()
        val_preds += outputs.tolist()
        
    val_loss /= len(val_loader.dataset)
    val_score = r2_score(val_labels, val_preds)
        
    return val_loss, val_score


def early_stopping():
    return True


def plot_losses(train_losses, val_losses):
    x = list(range(len(train_losses)))
    plt.plot(x, np.log10(train_losses), label='train')
    plt.plot(x, np.log10(val_losses), label='val')
    plt.legend()
    plt.show()
    plt.savefig('losses.png')
    plt.clf()


def train_model(epochs, model, optimizer, criterion, train_loader, val_loader):
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(epochs):
        print(f'\n--- EPOCH {epoch+1}/{epochs} ---')
        train_loss = train_epoch(model, optimizer, criterion, train_loader)
        train_losses.append(train_loss)
        
        val_loss, val_score = val_epoch(model, val_loader)
        val_losses.append(val_loss)
        
        print(f'Train (sqrt) = {sqrt(train_loss):.1f} - Val (sqrt) = {sqrt(val_loss):.1f} - Val R2 = {val_score:.2f}')
        
        plot_losses(train_losses, val_losses)

        
def round_prediction():
    return True


def make_submission(model, test_loader):
    print('\nCreate submission.csv')
    test_path = '../data/raw/test.csv'
    test_df = pd.read_csv(test_path)
    
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            keys_inputs = ['s_inputs', 'm_inputs', 'g_inputs']
            inputs = {key: data[key] for key in keys_inputs}

            district = data['district'][0]
            latitude = data['latitude'].item()
            longitude = data['longitude'].item()
            date_of_harvest = data['date_of_harvest'][0]

            output = model(inputs)

            test_df.loc[(test_df['District'] == district) &
                        (test_df['Latitude'] == latitude) &
                        (test_df['Longitude'] == longitude) &
                        (test_df['Date of Harvest'] == date_of_harvest),
                        'Predicted Rice Yield (kg/ha)'] = output.item()

    test_df.to_csv('submission.csv', index=False)
    
        
    
if __name__ == "__main__":
    epochs = 1000
    
    train_loader, val_loader, test_loader = get_loaders(batch_size=4, num_workers=4)
    first_batch = next(iter(train_loader))

    hidden_size = 32 # try 32, 64, 128
    num_layers = 1 # try 1, 2, 3, 4
    sequence_length = first_batch['s_inputs'].shape[1]
    s_num_features = first_batch['s_inputs'].shape[2]
    m_num_features = first_batch['m_inputs'].shape[2]
    g_in_features = first_batch['g_inputs'].shape[1]
    c_in_features = 1923

    model = DLModel(sequence_length, num_layers, hidden_size, s_num_features, m_num_features, g_in_features, c_in_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
        
    train_model(epochs, model, optimizer, criterion, train_loader, val_loader)
    make_submission(model, test_loader)