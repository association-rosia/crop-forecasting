import torch
import torch.nn as nn


class CustomModel(nn.Module):
    """ PyTorch class to define our custom model

    :param config: Contains all the model configs, also used in W&B logging
    :type config: dict
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
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

        self.lstm_p = config['lstm_dropout']
        self.fc_p = config['fc_dropout']

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
        self.lstm_dropout = nn.Dropout(self.lstm_p)
        self.fc_dropout = nn.Dropout(self.fc_p)

    def forward(self, x: dict) -> torch.Tensor:
        """ Model forward pass function

        :param x: Model inputs
        :type x: dict
        :return: Model output
        :rtype: torch.Tensor
        """
        s_input = x['s_input']
        m_input = x['m_input']
        g_input = x['g_input']

        # Spectral LSTM
        s_output, _ = self.s_lstm(s_input)
        s_output = self.s_bn_lstm(s_output[:, -1, :])
        s_output = self.tanh(s_output)
        s_output = self.lstm_dropout(s_output)

        # Spectral Conv1D
        s_output = torch.unsqueeze(s_output, 1)
        s_output = self.s_cnn(s_output)
        s_output = self.s_bn_cnn(torch.squeeze(s_output))
        s_output = self.relu(s_output)

        # Meteorological LSTM
        m_output, _ = self.m_lstm(m_input)
        m_output = self.m_bn_lstm(m_output[:, -1, :])
        m_output = self.tanh(m_output)
        m_output = self.lstm_dropout(m_output)

        # Meteorological Conv1D
        m_output = torch.unsqueeze(m_output, 1)
        m_output = self.m_cnn(m_output)
        m_output = self.m_bn_cnn(torch.squeeze(m_output))
        m_output = self.relu(m_output)

        # Concatenate inputs
        c_input = torch.cat((s_output, m_output, g_input), 1)
        c_output = self.c_bn_1(self.c_linear_1(c_input))
        c_output = self.relu(c_output)
        c_output = self.fc_dropout(c_output)
        c_output = self.c_bn_2(self.c_linear_2(c_output))
        c_output = self.relu(c_output)
        c_output = self.fc_dropout(c_output)
        output = self.c_linear_3(c_output)

        return output
