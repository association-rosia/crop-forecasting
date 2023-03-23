import warnings
warnings.filterwarnings('ignore')

import pytorch_lightning as pl
from model import LightningModel
from data import LightningData

import psutil

def objective(trial: optuna.trial.Trial) -> float:    
#     config = {
#         'batch_size': 16,  # try 8 to 128
#         's_hidden_size': 128,  # try 128 to 512
#         's_num_layers': 2,  # try 1 to 4
#         'm_hidden_size': 128,  # try 128 to 512
#         'm_num_layers': 2,  # try 1 to 4
#         'learning_rate': 1e-4,  # try 1e-5 to 1e-3
#         'dropout': 0.5,  # try 0.2 to 0.8
#         'epochs': 25,
#         'optimizer': 'AdamW',  # try AdamW and RMSprop
#         'criterion': 'MSELoss',  # try MSELoss, L1Loss and HuberLoss
#         'val_rate': 0.2,
#     }
    
    batch_size = trial.suggest_int('batch_size', 8, 128)
    
    data, first_batch = get_data(batch_size, val_rate=0.2)
    
    s_hidden_size = trial.suggest_int('s_hidden_size', 128, 256)
    s_num_layers = trial.suggest_int('s_num_layers', 1, 4)
    m_hidden_size = trial.suggest_int('m_hidden_size', 128, 256)
    m_num_layers = trial.suggest_int('m_num_layers', 1, 4)
    learning_rate = trial.suggest_uniform('learning_rate', 1e-5, 1e-3)
    # learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3) # ?????
    dropout = trial.suggest_uniform('dropout ', 0, 1)
    optimizer = trial.suggest_categorical('optimizer ', choices=['AdamW', 'RMSprop'])
    criterion = trial.suggest_categorical('criterion ', choices=['MSELoss', 'L1Loss', 'HuberLoss'])
    
    # ======== FIXED HYPERPARAMETERS ======== 
    config['train_size'] = len(data.train_dataset)
    config['val_size'] = len(data.val_dataset)
    config['s_num_features'] = first_batch['s_input'].shape[1]
    config['m_num_features'] = first_batch['m_input'].shape[1]
    config['g_in_features'] = first_batch['g_input'].shape[0]

    config['c_in_features'] = (config['s_hidden_size'] - 2 +
                               config['m_hidden_size'] - 2 +
                               config['g_in_features'])
    
    c_out_in_features_1 = trial.suggest_int('c_out_in_features_1', int(sqrt(config['c_in_features'])), 2*config['c_in_features'])
    c_out_in_features_2 = trial.suggest_int('c_out_in_features_2', int(sqrt(config['c_in_features'])), 2*config['c_in_features'])

    config['c_out_in_features_1'] = int(2 / 3 * config['c_in_features'])
    config['c_out_in_features_2'] = int(2 / 3 * config['c_in_features'])
    
    config = {
        'batch_size': 16,  # try 8 to 128
        's_hidden_size': 128,  # try 128 to 512
        's_num_layers': 2,  # try 1 to 4
        'm_hidden_size': 128,  # try 128 to 512
        'm_num_layers': 2,  # try 1 to 4
        'learning_rate': 1e-4,  # try 1e-5 to 1e-3
        'dropout': 0.5,  # try 0.2 to 0.8
        'epochs': 25,
        'optimizer': 'AdamW',  # try AdamW and RMSprop
        'criterion': 'MSELoss',  # try MSELoss, L1Loss and HuberLoss
        'val_rate': 0.2,
    }

    

    trainer = pl.Trainer(
        logger=True,
        max_epochs=25,
        accelerator='auto',
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_mean_r2_score')],
    )
    
    # hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
    # trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, data)

    return trainer.callback_metrics['val_mean_r2_score'].item()


# def main():
#     config = {
#         'batch_size': 16,  # try 8 to 128
#         's_hidden_size': 128,  # try 128 to 512
#         's_num_layers': 2,  # try 1 to 4
#         'm_hidden_size': 128,  # try 128 to 512
#         'm_num_layers': 2,  # try 1 to 4
#         'learning_rate': 1e-4,  # try 1e-5 to 1e-3
#         'dropout': 0.5,  # try 0.2 to 0.8
#         'epochs': 25,
#         'optimizer': 'AdamW',  # try AdamW and RMSprop
#         'criterion': 'MSELoss',  # try MSELoss, L1Loss and HuberLoss
#         'val_rate': 0.2,
#     }
    
#     config['train_size'] = len(data.train_dataset)
#     config['val_size'] = len(data.val_dataset)
#     config['s_num_features'] = first_batch['s_input'].shape[1]
#     config['m_num_features'] = first_batch['m_input'].shape[1]
#     config['g_in_features'] = first_batch['g_input'].shape[0]

#     config['c_in_features'] = (config['s_hidden_size'] - 2 +
#                                config['m_hidden_size'] - 2 +
#                                config['g_in_features'])

#     # try [sqrt(c_in_features) ; 2*c_in_features]
#     config['c_out_in_features_1'] = int(2 / 3 * config['c_in_features'])
#     config['c_out_in_features_2'] = int(2 / 3 * config['c_in_features'])

#     model = LightningModel(config)
#     trainer = pl.Trainer(logger=True)
#     trainer.fit(model, data)

#     # automatically loads the best weights for you
#     # trainer.test(model) # need to override the test_step() method
    
    
def get_data(batch_size, val_rate):
    num_workers = get_num_workers()
    data = LightningData(batch_size, val_rate, num_workers)
    data.setup(stage='fit')
    first_batch = data.train_dataset[0]
    
    return data, first_batch

    
def get_num_workers():
    if torch.backends.mps.is_available():
        num_workers = 0
    else:
        num_workers = psutil.cpu_count(logical=False)
        
    return num_workers


if __name__ == '__main__':
    main()
