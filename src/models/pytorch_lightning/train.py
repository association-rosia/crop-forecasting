from model import LightningModel


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

    model = LightningModel(config)


if __name__ == '__main__':
    main()
