import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
import pandas as pd

def train_model(model,train_dl,optimizer,criterion):
    model.train()
    
    epoch_loss = 0
    epoch_acc = 0
    
    for x, y in train_dl:
        y = y.type(torch.float32)
        y_pred = model(x).squeeze(-1)


        optimizer.zero_grad()

#             loss = F.cross_entropy(y_pred, y)
        loss = criterion(y_pred,y)
        loss.backward()
        optimizer.step()
        
        acc = binary_accuracy(y_pred, y)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(train_dl), epoch_acc / len(train_dl)

def evaluate (model, valid_dl):
    model.eval()
    epoch_acc = 0
    with torch.no_grad():
        for x, y in valid_dl:
            y = y.type(torch.float32)
            y_hat = model(x).squeeze(-1)
            acc = binary_accuracy(y_hat,y)

            epoch_acc += acc.item()
#             print(acc.item())
            
    return epoch_acc / len(valid_dl)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc