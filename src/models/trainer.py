import torch 
from tqdm import tqdm
from sklearn.metrics import r2_score
import pandas as pd
from datetime import datetime
from os.path import join
import wandb

import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from utils import ROOT_DIR

class Trainer():
    def __init__(self, model, train_loader, val_loader, epochs, criterion, optimizer, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device        
        self.timestamp = int(datetime.now().timestamp())
        self.best_score = 0.6

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
            
            pbar.set_description(f'TRAIN - Batch: {i + 1}/{len(self.train_loader)} Epoch Loss: {epoch_loss:.5f} - Batch Loss: {loss.item():.5f}')
            
        train_loss /= len(self.train_loader)
        
        return train_loss
    
    def compute_r2_scores(self, observations, labels, preds):
        df = pd.DataFrame()
        df['observations'] = observations
        df['labels'] = labels
        df['preds'] = preds
        full_r2_score = r2_score(df.labels, df.preds)
        df = df.groupby(['observations']).mean()
        mean_r2_score = r2_score(df.labels, df.preds)

        return full_r2_score, mean_r2_score

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
                
            pbar.set_description(f'VAL - Batch: {i + 1}/{len(self.val_loader)} Epoch Loss: {epoch_loss:.5f} - Batch Loss: {loss.item():.5f}')
            
        val_loss /= len(self.val_loader)
        val_r2_score, val_mean_r2_score = self.compute_r2_scores(observations, val_labels, val_preds)
            
        return val_loss, val_r2_score, val_mean_r2_score

    def early_stopping(self):
        return True
    
    def save(self, score):
        save_folder = join(ROOT_DIR, 'models') 
        
        if score > self.best_score:
            self.best_score = score
            os.makedirs(save_folder, exist_ok=True)
            
            # delete former best model 
            former_model = [f for f in os.listdir(save_folder) if f.split('_')[0] == str(self.timestamp)]
            if len(former_model) == 1:
                os.remove(join(save_folder, former_model[0]))
            
            # save new model 
            score = str(score)[:7].replace('.', '-')
            file_name = f'{self.timestamp}_model_{score}.pt'
            save_path = join(save_folder, file_name)
            torch.save(self.model, save_path)
        
    def train(self): # train model 
        train_losses = []
        val_losses = []

        iter_epoch = tqdm(range(self.epochs), leave=False)
        for epoch in iter_epoch:
            iter_epoch.set_description(f'EPOCH {epoch+1}/{self.epochs}')
            train_loss = self.train_one_epoch()
            train_losses.append(train_loss)

            val_loss, val_r2_score, val_mean_r2_score = self.val_one_epoch()
            self.scheduler.step(val_loss)
            val_losses.append(val_loss)
            
            self.save(val_mean_r2_score)

            wandb.log({'train_loss': train_loss, 
                       'val_loss': val_loss, 
                       'val_r2_score': val_r2_score, 
                       'val_mean_r2_score': val_mean_r2_score,
                       'best_val_mean_r2_score': self.best_score
                      })
            
            iter_epoch.write(f'EPOCH {epoch + 1}/{self.epochs}: Train = {train_loss:.5f} - Val = {val_loss:.5f} - Val R2 = {val_r2_score:.5f} - Val mean R2 = {val_mean_r2_score:.5f}')