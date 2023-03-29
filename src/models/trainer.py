import os
import sys
from datetime import datetime
from math import sqrt
from os.path import join

import pandas as pd
import torch
from sklearn.metrics import r2_score
from tqdm import tqdm

import wandb

parent = os.path.abspath('.')
sys.path.insert(1, parent)
from utils import ROOT_DIR


def compute_r2_scores(observations, labels, preds):
    df = pd.DataFrame()
    df['observations'] = observations
    df['labels'] = labels
    df['preds'] = preds
    full_r2_score = r2_score(df.labels, df.preds)
    df = df.groupby(['observations']).mean()
    mean_r2_score = r2_score(df.labels, df.preds)

    return full_r2_score, mean_r2_score


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, epochs, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.criterion = criterion
        self.epochs = epochs
        self.optimizer = optimizer
        self.device = device
        self.timestamp = int(datetime.now().timestamp())
        self.val_best_r2_score = 0.

    def train_one_epoch(self):
        train_loss = 0.

        self.model.train()

        pbar = tqdm(self.train_loader, leave=False)
        for i, data in enumerate(pbar):
            keys_input = ['s_input', 'm_input', 'g_input']
            inputs = {key: data[key].to(self.device) for key in keys_input}
            labels = data['target'].float().to(self.device)

            # Zero gradients for every batch
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            train_loss += loss.item()
            epoch_loss = train_loss / (i + 1)

            pbar.set_description(f'TRAIN - Batch: {i + 1}/{len(self.train_loader)} - '
                                 f'Epoch Loss: {epoch_loss:.5f} - '
                                 f'Batch Loss: {loss.item():.5f}')

        train_loss /= len(self.train_loader)

        return train_loss

    def val_one_epoch(self):
        val_loss = 0.
        observations = []
        val_labels = []
        val_preds = []

        self.model.eval()

        pbar = tqdm(self.val_loader, leave=False)
        for i, data in enumerate(pbar):
            keys_input = ['s_input', 'm_input', 'g_input']
            inputs = {key: data[key].to(self.device) for key in keys_input}
            labels = data['target'].float().to(self.device)

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            val_loss += loss.item()
            epoch_loss = val_loss / (i + 1)

            observations += data['observation'].squeeze().tolist()
            val_labels += labels.squeeze().tolist()
            val_preds += outputs.squeeze().tolist()

            pbar.set_description(f'VAL - Batch: {i + 1}/{len(self.val_loader)} - '
                                 f'Epoch Loss: {epoch_loss:.5f} - '
                                 f'Batch Loss: {loss.item():.5f}')

        val_loss /= len(self.val_loader)
        val_r2_score, val_mean_r2_score = compute_r2_scores(observations, val_labels, val_preds)

        return val_loss, val_r2_score, val_mean_r2_score

    def save(self, score):
        save_folder = join(ROOT_DIR, 'models')

        if score > self.val_best_r2_score:
            self.val_best_r2_score = score
            os.makedirs(save_folder, exist_ok=True)

            # delete former best model 
            former_model = [f for f in os.listdir(save_folder) if f.split('_')[-1] == f'{self.timestamp}.pt']
            if len(former_model) == 1:
                os.remove(join(save_folder, former_model[0]))

            # save new model 
            score = str(score)[:7].replace('.', '-')
            file_name = f'{score}_model_{self.timestamp}.pt'
            save_path = join(save_folder, file_name)
            torch.save(self.model, save_path)

    def train(self):  # train model
        iter_epoch = tqdm(range(self.epochs), leave=False)

        for epoch in iter_epoch:
            iter_epoch.set_description(f'EPOCH {epoch + 1}/{self.epochs}')
            train_loss = self.train_one_epoch()

            val_loss, val_r2_score, val_mean_r2_score = self.val_one_epoch()
            self.save(val_mean_r2_score)

            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_r2_score': val_r2_score,
                'val_mean_r2_score': val_mean_r2_score,
                'val_best_r2_score': self.val_best_r2_score
            })

            iter_epoch.write(f'EPOCH {epoch + 1}/{self.epochs}: '
                             f'Train = {train_loss:.5f} - '
                             f'Val = {val_loss:.5f} - '
                             f'Val R2 = {val_r2_score:.5f} - '
                             f'Val mean R2 = {val_mean_r2_score:.5f}')
