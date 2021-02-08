import torch
import torch.nn as nn

class LSTM_fixed_len(torch.nn.Module) :
    def __init__(self,vocab_size,embedding_dim, hidden_dim,num_layers,bidirectional,dropout,n_class) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim,num_layers = 12, batch_first=True)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
#         self.linear = nn.Linear(hidden_dim,n_class)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        
#         self.fc2 = nn.Linear(hidden_dim, n_class)
        self.fc2 = nn.Linear(hidden_dim, n_class)
        
        self.dropout = nn.Dropout(dropout)
#         self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.embeddings(x)
#         x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
    
        hidden = self.dropout(torch.cat((ht[-2], ht[-1]), dim = 1))
        
        output = self.fc1(hidden)
        
        output = self.dropout(self.fc2(output))
#         output = self.fc2(output)
        
        return output