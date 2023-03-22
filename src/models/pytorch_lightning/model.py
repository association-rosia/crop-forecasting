import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score
import pandas as pd
from datetime import datetime

import os
import sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)
from utils import ROOT_DIR


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.s_hidden_size = config['s_hidden_size']
        self.m_hidden_size = config['m_hidden_size']
        self.s_num_features = config['s_num_features']

        self.s_num_layers = config['s_num_layers']
        self.m_num_layers = config['m_num_layers']
        self.m_num_features = config['m_num_features']

        self.g_in_features = config['g_in_features']

        self.c_in_features = config['c_in_features']
        self.c_out_in_features_1 = config['c_out_in_features_1']
        self.c_out_in_features_2 = config['c_out_in_features_2']

        self.dropout = config['dropout']

        self.s_lstm = nn.LSTM(self.s_num_features, self.s_hidden_size, self.s_num_layers, batch_first=True)
        self.s_bn_lstm = nn.BatchNorm1d(self.s_hidden_size)
        self.s_cnn = nn.Conv1d(1, 1, kernel_size=3)
        self.s_bn_cnn = nn.BatchNorm1d(self.s_hidden_size - 2)  # because kernel_size = 3

        self.m_lstm = nn.LSTM(self.m_num_features, self.m_hidden_size, self.m_num_layers, batch_first=True)
        self.m_bn_lstm = nn.BatchNorm1d(self.m_hidden_size)
        self.m_cnn = nn.Conv1d(1, 1, kernel_size=3)
        self.m_bn_cnn = nn.BatchNorm1d(self.m_hidden_size - 2)

        self.c_linear_1 = nn.Linear(self.c_in_features, self.c_out_in_features_1)
        self.c_bn_1 = nn.BatchNorm1d(self.c_out_in_features_1)
        self.c_linear_2 = nn.Linear(self.c_out_in_features_1, self.c_out_in_features_2)
        self.c_bn_2 = nn.BatchNorm1d(self.c_out_in_features_2)
        self.c_linear_3 = nn.Linear(self.c_out_in_features_2, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        s_input = x['s_input']
        m_input = x['m_input']
        g_input = x['g_input']

        # Spectral LSTM
        s_h0 = torch.zeros(self.s_num_layers, s_input.size(0), self.s_hidden_size).requires_grad_()
        s_c0 = torch.zeros(self.s_num_layers, s_input.size(0), self.s_hidden_size).requires_grad_()
        s_output, _ = self.s_lstm(s_input, (s_h0, s_c0))
        s_output = self.s_bn_lstm(s_output[:, -1, :])
        s_output = self.tanh(s_output)
        s_output = self.dropout(s_output)

        # Spectral Conv1D
        s_output = torch.unsqueeze(s_output, 1)
        s_output = self.s_cnn(s_output)
        s_output = self.s_bn_cnn(torch.squeeze(s_output))
        s_output = self.relu(s_output)
        s_output = self.dropout(s_output)

        # Meteorological LSTM
        m_h0 = torch.zeros(self.m_num_layers, m_input.size(0), self.m_hidden_size).requires_grad_()
        m_c0 = torch.zeros(self.m_num_layers, m_input.size(0), self.m_hidden_size).requires_grad_()
        m_output, _ = self.m_lstm(m_input, (m_h0, m_c0))
        m_output = self.m_bn_lstm(m_output[:, -1, :])
        m_output = self.tanh(m_output)
        m_output = self.dropout(m_output)

        # Meteorological Conv1D
        m_output = torch.unsqueeze(m_output, 1)
        m_output = self.m_cnn(m_output)
        m_output = self.m_bn_cnn(torch.squeeze(m_output))
        m_output = self.relu(m_output)
        m_output = self.dropout(m_output)

        # Concatenate inputs
        c_input = torch.cat((s_output, m_output, g_input), 1)
        c_output = self.c_bn_1(self.c_linear_1(c_input))
        c_output = self.relu(c_output)
        c_output = self.dropout(c_output)
        c_output = self.c_bn_2(self.c_linear_2(c_output))
        c_output = self.relu(c_output)
        c_output = self.dropout(c_output)
        output = self.c_linear_3(c_output)

        return output


class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super(LightningModel, self).__init__()
        self.model = Model(config)
        self.lr = config['learning_rate']
        self.keys_input = ['s_input', 'm_input', 'g_input']
        self.timestamp = int(datetime.now().timestamp())
        self.best_score = 0
        self.val_observations = []
        self.val_labels = []
        self.val_outputs = []

    def criterion(self, outputs, labels):
        return nn.MSELoss(outputs, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs = {key: train_batch[key] for key in self.keys_input}

        outputs = self.model(inputs)

        labels = train_batch['target'].float()

        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs = {key: val_batch[key] for key in self.keys_input}
        self.val_observations += val_batch['observation'].squeeze().tolist()

        outputs = self.model(inputs)
        self.val_preds += outputs.squeeze().tolist()

        labels = val_batch['target'].float()
        self.val_labels += labels.squeeze().tolist()

        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss.item())
        return loss

    def r2_scores(self):
        df = pd.DataFrame()
        df['observations'] = self.val_observations
        df['outputs'] = self.val_outputs
        df['labels'] = self.val_labels
        full_r2_score = r2_score(df.labels, df.outputs)
        df = df.groupby(['observations']).mean()
        mean_r2_score = r2_score(df.labels, df.outputs)

        return full_r2_score, mean_r2_score

    def save_model(self, score):
        save_folder = os.path.join(ROOT_DIR, 'models')

        if score > self.best_score:
            self.best_score = score
            os.makedirs(save_folder, exist_ok=True)

            former_model = [f for f in os.listdir(save_folder) if f.split('_')[0] == str(self.timestamp)]
            if len(former_model) == 1:
                os.remove(os.path.join(save_folder, former_model[0]))

            score = str(score)[:7].replace('.', '-')
            file_name = f'{self.timestamp}_model_{score}.pt'
            save_path = os.path.join(save_folder, file_name)
            torch.save(self.model, save_path)

    def on_validation_epoch_end(self):
        val_r2_score, val_mean_r2_score = self.r2_scores()
        self.log('val_r2_score', val_r2_score)
        self.log('val_mean_r2_score', val_mean_r2_score)
        self.val_observations.clear()
        self.val_outputs.clear()
        self.val_labels.clear()
