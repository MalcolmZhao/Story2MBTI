#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

# In[2]:


import os

# In[ ]:


log_name = "20240429213105_log.pkl"
log_path = os.path.join("results", log_name)

# In[ ]:


import pickle
from torch.utils.data import DataLoader, Dataset
class TextDataset(Dataset):
    def __init__(self, X, mask, y):
        self.X = X
        self.mask = mask
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]
# In[ ]:


model_log = pickle.load(open(log_path, "rb"))

# In[ ]:


from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

# In[ ]:


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cuda detected")
else:
    device = torch.device("cpu")
    print("Running on CPU")

# In[ ]:


from mbti_bert_model import mbti_bert

# In[ ]:


model_name = 'bert-large-uncased'  # Choose the pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
model = mbti_bert(bert_model).to(device)

# In[ ]:


model_weight_name = "20240429213105_model.pth"
model_weight_path = os.path.join("variables", model_weight_name)
model.load_state_dict(torch.load(model_weight_path))

# In[ ]:


X_val, y_val, mask_val = model_log[3], model_log[4], model_log[5]
X_val, y_val, mask_val = X_val.to(device), y_val.to(device), mask_val.to(device)
X_test, y_test, mask_test = model_log[6], model_log[7], model_log[8]
X_test, y_test, mask_test = X_test.to(device), y_test.to(device), mask_test.to(device)

# In[1]:


# define a function to iterate through validation set to find the optimal benchmark using f1-score
from sklearn.metrics import f1_score
import numpy as np

# In[ ]:

# use validation set, f1 score to find the optimal benchmark to classify the channels
def calculate_f1_score(y_pred, y, benchmark):
    y_hat = np.array(y_pred >= benchmark, dtype = int)
    y_label = np.array(y >= 0.5, dtype = int)
    f1 = f1_score(y_label, y_hat, average = 'binary')
    return f1

# In[ ]:


def optimal_single_channle(score_range, y_pred, y):
    f1_scores = []
    for score in score_range:
        f1 = calculate_f1_score(y_pred, y, score)
        f1_scores.append(f1)
    f1_scores = np.array(f1_scores)
    return score_range[np.argmax(f1_scores)]

# In[ ]:


def find_optimal_benchmark(score_range, model, X_val, mask_val, y_val):
    model.eval()
    y_pred = model(X_val, mask_val)
    y_pred = [item.cpu().detach().numpy() for item in y_pred]
    y_val = y_val.cpu().detach().numpy()
    ans = []
    for i in range(4):
        cur_benchmark = optimal_single_channle(score_range, y_pred[i], y_val[:, i])
        ans.append(cur_benchmark)
    return ans

# In[ ]:


benchmark = find_optimal_benchmark(np.arange(0.1, 0.9, 0.01), model, X_val, mask_val, y_val)

# In[ ]:


f1_scores = []
model.eval()
y_pred = model(X_test, mask_test)
y_pred = [item.cpu().detach().numpy() for item in y_pred]
y_test = y_test.cpu().detach().numpy()
for i in range(4):
    f1 = calculate_f1_score(y_pred[i], y_test[:, i], benchmark[i])
    f1_scores.append(f1)
print("Test set result:", f1_scores)
# Test set result: [0.6477987421383647 (i), 0.5808580858085808 (n), 0.7240356083086054 (f), 0.7048192771084337 (p)]