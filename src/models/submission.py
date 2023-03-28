import os
import sys

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

parent = os.path.abspath('.')
sys.path.insert(1, parent)

from os.path import join

from dataloader import get_dataloaders

from src.constants import TARGET, TARGET_TEST
from utils import ROOT_DIR

MODEL = 'lucky-sweep-4.pt'


def rounded_yield(x, crop_yields):
    diffs = [abs(x - crop_yield) for crop_yield in crop_yields]
    return crop_yields[diffs.index(min(diffs))]


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        raise Exception("None accelerator available")

    return device


class Evaluator:
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
        
        test_df[TARGET_TEST] = scaler.inverse_transform(df[['preds']])
        
        crop_yields = train_df[TARGET].unique().tolist()
        test_df[TARGET_TEST] = test_df[TARGET_TEST].apply(lambda x: rounded_yield(x, crop_yields))
        test_df.to_csv(f'submissions/{MODEL}.csv', index=False)
        
        return True
        
    def evaluate(self, model):
        observations = []
        test_preds = []

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
    device = get_device()
    _, _, test_dataloader = get_dataloaders(batch_size=64, val_rate=0.2)
    
    evaluator = Evaluator(test_dataloader, device)
    
    model_path = join(ROOT_DIR, 'models', MODEL)
    model = torch.load(model_path).to(device)
    evaluator.evaluate(model)
    