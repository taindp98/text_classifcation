import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
import pandas as pd



def train_model(model,train_dl,optimizer,criterion,writer,epoch):
    model.train()
    
    epoch_loss = 0
    epoch_acc = 0
    list_pred = []
    for x, y in train_dl:
        y = y.type(torch.int64)
        x = x.long()
#         print('y',y)
#         print('y',y.dtype)

        y_pred = model(x)
#         print('y_pred',y_pred)
#         print('y_pred',y_pred.dtype)
        optimizer.zero_grad()


        loss = criterion(y_pred,y)
        
        writer.add_scalar("Loss/train", loss, epoch)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
#         epoch_acc += acc.item()
        list_pred.append(y_pred.argmax())
#     return epoch_loss / len(train_dl), epoch_acc / len(train_dl)
    return epoch_loss / len(train_dl)

def evaluate (model, valid_dl,criterion,writer,epoch):
    model.eval()
    epoch_acc = 0
    epoch_loss = 0
    list_true = []
    list_pred = []
    with torch.no_grad():
        for x, y in valid_dl:
            y = y.type(torch.int64)
            x = x.long()
            y_hat = model(x)
#             acc = binary_accuracy(y_hat,y)
            loss = criterion(y_hat,y)
            writer.add_scalar("Loss/valid", loss, epoch)

            epoch_loss += loss.item()
#             epoch_acc += acc.item()
#             print(acc.item())
#             predlist=torch.cat([predlist,preds.view(-1).cpu()])
#             y_hat_round = torch.round(torch.sigmoid(y_hat))
            
#             list_true.append(y)
#             list_pred.append(y_hat_round)
#     return epoch_acc
    return epoch_loss/len(valid_dl)
#     return epoch_acc / len(valid_dl),list_true,list_pred

# def binary_accuracy(preds, y):
#     """
#     Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
#     """

#     #round predictions to the closest integer
# #     rounded_preds = torch.round(torch.sigmoid(preds))
#     pred_argmax = preds.argmax()
#     y_squeeze = y.squeeze()
#     correct = (pred_argmax == y).float() #convert into float for division 
#     acc = correct.sum() / len(correct)
#     return acc

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model