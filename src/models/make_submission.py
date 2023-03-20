import pandas as pd
import torch
from tqdm import tqdm

import xarray as xr

from sklearn.preprocessing import MinMaxScaler

import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from dataloader import get_loaders

from src.constants import FOLDER, TARGET

from utils import ROOT_DIR
from os.path import join


class Evaluator():
    def __init__(self, test_loader, device):
        self.test_loader = test_loader
        self.device = device
    
    def create_submission(self, observations, preds):
        df = pd.DataFrame()
        df['observations'] = observations
        df['preds'] = preds
        df = df.groupby(['observations']).mean()
        df = df.sort_values(by='observations')

        test_path = join(ROOT_DIR, 'data', 'raw', 'test.csv')
        test_df = pd.read_csv(test_path)
        
        scaler = MinMaxScaler()
        train_path = join(ROOT_DIR, 'data', 'raw', 'train.csv')
        train_df = pd.read_csv(train_path)
        scaler.fit(train_df[[TARGET]])
        
        test_df['Predicted Rice Yield (kg/ha)'] = scaler.inverse_transform(df[['preds']])
        test_df['Predicted Rice Yield (kg/ha)'] = test_df['Predicted Rice Yield (kg/ha)'].apply(lambda x: int(x/10)*10)
        test_df.to_csv('submission.csv')
        
    def evaluate(self, model):
        observations = []
        test_preds = []
        
        model.eval()

        pbar = tqdm(self.test_loader, leave=False)
        for i, data in enumerate(pbar):
            keys_input = ['s_input', 'm_input', 'g_input']
            inputs = {key: data[key].to(self.device) for key in keys_input}
            outputs = model(inputs)
            
            observations += data['observation'].squeeze().tolist()
            test_preds += outputs.squeeze().tolist()
                
            pbar.set_description(f'TEST - Batch: {i + 1}/{len(self.test_loader)}')
            
        self.create_submission(observations, test_preds)
        
        
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = {'batch_size': 8, 'val_rate': 0.2}
    _, _, test_loader = get_loaders(config, num_workers=4)
    
    evaluator = Evaluator(test_loader, device)
    
    model_path = join(ROOT_DIR, 'models', '1679248975_model_0-62500.pt')
    model = torch.load(model_path)
    evaluator.evaluate(model)
    