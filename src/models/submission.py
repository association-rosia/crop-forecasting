from string import Template
import argparse, os

import pandas as pd

import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

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
        
        test_df['Predicted Rice Yield (kg/ha)'] = df['preds']
        test_df.to_csv('submission.csv')
        
    def evaluate(self, model):
        observations = []
        test_preds = []
        
        model.eval()

        pbar = tqdm(self.test_loader, leave=False)
        for i, data in enumerate(pbar):
            keys_input = ['s_input', 'm_input', 'g_input']
            inputs = {key: data[key].to(self.device) for key in keys_input}
            labels = data['target'].float().to(self.device)

            outputs = model(inputs)            
            observations += data['observation'].squeeze().tolist()
            test_preds += outputs.squeeze().tolist()
                
            pbar.set_description(f'TEST - Batch: {i + 1}/{len(self.val_loader)}')
            
        self.create_submission(observations, test_preds)