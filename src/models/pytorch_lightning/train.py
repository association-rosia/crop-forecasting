import warnings
warnings.filterwarnings('ignore')

import pytorch_lightning as pl
from model import LightningModel
from data import LightningData


def main():
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

    data = LightningData(config, num_workers=4)
    data.setup(stage='fit')
    first_batch = data.train_dataset[0]

    config['train_size'] = len(data.train_dataset)
    config['val_size'] = len(data.val_dataset)
    config['s_num_features'] = first_batch['s_input'].shape[1]
    config['m_num_features'] = first_batch['m_input'].shape[1]
    config['g_in_features'] = first_batch['g_input'].shape[0]

    config['c_in_features'] = (config['s_hidden_size'] - 2 +
                               config['m_hidden_size'] - 2 +
                               config['g_in_features'])

    # try [sqrt(c_in_features) ; 2*c_in_features]
    config['c_out_in_features_1'] = int(2 / 3 * config['c_in_features'])
    config['c_out_in_features_2'] = int(2 / 3 * config['c_in_features'])

    model = LightningModel(config)
    trainer = pl.Trainer(logger=True)
    trainer.fit(model, data)

    # automatically loads the best weights for you
    # trainer.test(model) # need to override the test_step() method


if __name__ == '__main__':
    main()
