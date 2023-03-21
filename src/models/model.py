import torch
import torch.nn as nn 

class LSTMModel(nn.Module):
    def __init__(self, config, device):
        super(LSTMModel, self).__init__()
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']

        self.s_num_features = config['s_num_features']
        self.m_num_features = config['m_num_features']
        self.g_in_features = config['g_in_features']
        self.c_in_features = config['c_in_features']
        self.c_out_in_features_1 = config['c_out_in_features_1']
        self.c_out_in_features_2 = config['c_out_in_features_2']
        
        self.device = device
        
        self.s_lstm = nn.LSTM(self.s_num_features, self.hidden_size, self.num_layers, batch_first=True)
        self.m_lstm = nn.LSTM(self.m_num_features, self.hidden_size, self.num_layers, batch_first=True)
        self.bn_lstm = nn.BatchNorm1d(self.hidden_size)
        
        self.cnn = nn.Conv1d(1, 1, kernel_size=3)
        self.bn_cnn = nn.BatchNorm1d(self.hidden_size - 2) # because kernel_size = 3
        
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
        s_h0 = torch.zeros(self.num_layers, s_input.size(0), self.hidden_size).requires_grad_().to(self.device)
        s_c0 = torch.zeros(self.num_layers, s_input.size(0), self.hidden_size).requires_grad_().to(self.device)
        
        s_output, _ = self.s_lstm(s_input, (s_h0, s_c0))        
        s_output = self.bn_lstm(s_output[:, -1, :])
        s_output = self.tanh(s_output)
        s_output = self.dropout(s_output)
        
        # Meteo LSTM
        m_h0 = torch.zeros(self.num_layers, m_input.size(0), self.hidden_size).requires_grad_().to(self.device)
        m_c0 = torch.zeros(self.num_layers, m_input.size(0), self.hidden_size).requires_grad_().to(self.device)
        m_output, _ = self.m_lstm(m_input, (m_h0, m_c0))        
        m_output = self.bn_lstm(m_output[:, -1, :])
        m_output = self.tanh(m_output)
        m_output = self.dropout(m_output)
        
        # Spectral Conv1D
        s_output = torch.unsqueeze(s_output, 1) # (batch_size, num_layers) to (batch_size, 1, num_layers)        
        s_output = self.cnn(s_output)
        s_output = self.bn_cnn(torch.squeeze(s_output)) # (batch_size, 1, num_layers - 2) to (batch_size, num_layers - 2)           
        s_output = self.relu(s_output)
        s_output = self.dropout(s_output)
        
        # Meteo Conv1D
        m_output = torch.unsqueeze(m_output, 1)    
        m_output = self.cnn(m_output)
        m_output = self.bn_cnn(torch.squeeze(m_output))    
        m_output = self.relu(m_output)
        m_output = self.dropout(m_output)
        
        # Concatanate inputs
        c_input = torch.cat((s_output, m_output, g_input), 1)
        c_output = self.c_bn_1(self.c_linear_1(c_input))
        c_output = self.relu(c_output)
        c_output = self.dropout(c_output)
        c_output = self.c_bn_2(self.c_linear_2(c_output))
        c_output = self.relu(c_output)
        c_output = self.dropout(c_output)
        output = self.c_linear_3(c_output)
        
        return output