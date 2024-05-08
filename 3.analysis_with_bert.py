#!/usr/bin/env python
# coding: utf-8

# In[137]:


import warnings
warnings.filterwarnings("ignore")

# In[138]:


import numpy as np
import pandas as pd
import random

# In[143]:


seed = 434
raw_data = pd.read_excel("data/MVMR_BERT_tok.xlsx")
# raw_data = raw_data.iloc[0:200, :]

# In[144]:


raw_data[["i_score", "n_score", "f_score", "p_score"]] = raw_data[["i_score", "n_score", "f_score", "p_score"]] / 100

# In[146]:

# one model is to do classification so labels are needed
raw_data["i_label"] = raw_data["i_score"].apply(lambda x: 1 if x < 0.5 else 0)
raw_data["n_label"] = raw_data["n_score"].apply(lambda x: 1 if x < 0.5 else 0)
raw_data["f_label"] = raw_data["f_score"].apply(lambda x: 1 if x < 0.5 else 0)
raw_data["p_label"] = raw_data["p_score"].apply(lambda x: 1 if x < 0.5 else 0)

# In[147]:


random.seed(seed)
np.random.seed(seed)

# In[148]:


import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# In[149]:

# this "large" model is used for both classfication and continuous score prediction
from transformers import BertTokenizer, BertModel
model_name = 'bert-large-uncased'  # one model named "bert-base-uncased" is also used for training
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# In[150]:


def yield_tokens(data):
    for text in data:
        yield tokenizer.tokenize(text)

# In[151]:


from collections import Counter
counter = Counter()
for tokens in yield_tokens(raw_data["dialog"]):
    counter.update(tokens)

vocab = build_vocab_from_iterator(yield_tokens(raw_data["dialog"]), specials = ["<unk>", "<pad>", "<bos>", "<eos>"])
print(f"Vocab size: {len(vocab)}")

# In[201]:

max_length = 512 # for the restriction of BERT model
def text_pipeline(text, tokenizer, vocab):
    tokenized_text = tokenizer.tokenize(text)
    tokenized_text = ["[CLS]"] + tokenized_text + ["[SEP]"] # for classification purpose, otherwise the bert model will predict encodings for sequence generation
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    return torch.tensor(indexed_tokens, dtype=torch.long)

def value_pipeline(target):
    return torch.tensor(target, dtype = torch.float32)

def label_pipeline(target):
    return torch.tensor([target, 1-target], dtype = torch.float32)

# In[202]:


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cuda detected")
else:
    device = torch.device("cpu")
    print("Running on CPU")

# In[203]:


# X = [text_pipeline(dialog, tokenizer, vocab) for dialog in raw_data["dialog"]]
X = [torch.tensor(tokenizer.encode(dialog, max_length=512, truncation=True)) for dialog in raw_data["dialog"]]
y = [value_pipeline(target) for target in raw_data[["i_score", "n_score", "f_score", "p_score"]].values.T]
i_label = [label_pipeline(label) for label in raw_data["i_label"].values]
n_label = [label_pipeline(label) for label in raw_data["n_label"].values]
f_label = [label_pipeline(label) for label in raw_data["f_label"].values]
p_label = [label_pipeline(label) for label in raw_data["p_label"].values]
y_label = [torch.stack([i, n, f, p], dim = 0) for i, n, f, p in zip(i_label, n_label, f_label, p_label)]

from torch.nn.utils.rnn import pad_sequence
X = pad_sequence(X, batch_first = True, padding_value = tokenizer.pad_token_id).to(device)
y = torch.stack(y, dim = 1).to(device)
y_label = torch.stack(y_label, dim = 0).to(device)

# In[204]:


def create_attention_mask(input_ids):
    # Create attention mask tensor with shape (batch_size, max_seq_length)
    attention_mask = (input_ids != 0).float()  # Masking token (token ID 0) is set to 0, others to 1
    return attention_mask.to(device)

# In[205]:


from sklearn.model_selection import train_test_split
X_train, X_test_tmp, y_train, y_test_tmp = train_test_split(X, y, test_size = 0.2)
X_val, X_test, y_val, y_test = train_test_split(X_test_tmp, y_test_tmp, test_size = 0.5)
mask_train = create_attention_mask(X_train)
mask_val = create_attention_mask(X_val)
mask_test = create_attention_mask(X_test)
print(f"Train size: {len(X_train)}")
print(f"Valid size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# In[206]:


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

# In[207]:


train_dataset = TextDataset(X_train, mask_train, y_train)
valid_dataset = TextDataset(X_val, mask_val, y_val)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
val_loader = DataLoader(valid_dataset, batch_size = 64, shuffle = True)

# #### Hugging Face Bert

# In[208]:


import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup


# In[209]:


class mbti_bert(nn.Module):
    def __init__(self, bert_model):
        super(mbti_bert, self).__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in bert_model.encoder.layer[-4:].parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.1)
        self.i_output_1 = nn.Linear(self.bert.config.hidden_size, 1024)
        self.bni_1 = nn.BatchNorm1d(1024)
        self.i_output_2 = nn.Linear(1024, 512)
        self.bni_2 = nn.BatchNorm1d(512)
        self.i_output_3 = nn.Linear(512, 512)
        self.bni_3 = nn.BatchNorm1d(512)
        self.i_output_4 = nn.Linear(512, 128)
        self.bni_4 = nn.BatchNorm1d(128)
        self.i_output_5 = nn.Linear(128, 64)
        self.bni_5 = nn.BatchNorm1d(64)
        self.i_output_6 = nn.Linear(64, 16)
        self.bni_6 = nn.BatchNorm1d(16)
        self.i_output_7 = nn.Linear(16, 1)

        self.n_output_1 = nn.Linear(self.bert.config.hidden_size, 1024)
        self.bnn_1 = nn.BatchNorm1d(1024)
        self.n_output_2 = nn.Linear(1024, 512)
        self.bnn_2 = nn.BatchNorm1d(512)
        self.n_output_3 = nn.Linear(512, 512)
        self.bnn_3 = nn.BatchNorm1d(512)
        self.n_output_4 = nn.Linear(512, 128)
        self.bnn_4 = nn.BatchNorm1d(128)
        self.n_output_5 = nn.Linear(128, 64)
        self.bnn_5 = nn.BatchNorm1d(64)
        self.n_output_6 = nn.Linear(64, 16)
        self.bnn_6 = nn.BatchNorm1d(16)
        self.n_output_7 = nn.Linear(16, 1)

        self.f_output_1 = nn.Linear(self.bert.config.hidden_size, 1024)
        self.bnf_1 = nn.BatchNorm1d(1024)
        self.f_output_2 = nn.Linear(1024, 512)
        self.bnf_2 = nn.BatchNorm1d(512)
        self.f_output_3 = nn.Linear(512, 512)
        self.bnf_3 = nn.BatchNorm1d(512)
        self.f_output_4 = nn.Linear(512, 128)
        self.bnf_4 = nn.BatchNorm1d(128)
        self.f_output_5 = nn.Linear(128, 64)
        self.bnf_5 = nn.BatchNorm1d(64)
        self.f_output_6 = nn.Linear(64, 16)
        self.bnf_6 = nn.BatchNorm1d(16)
        self.f_output_7 = nn.Linear(16, 1)

        self.p_output_1 = nn.Linear(self.bert.config.hidden_size, 1024)
        self.bnp_1 = nn.BatchNorm1d(1024)
        self.p_output_2 = nn.Linear(1024, 512)
        self.bnp_2 = nn.BatchNorm1d(512)
        self.p_output_3 = nn.Linear(512, 512)
        self.bnp_3 = nn.BatchNorm1d(512)
        self.p_output_4 = nn.Linear(512, 128)
        self.bnp_4 = nn.BatchNorm1d(128)
        self.p_output_5 = nn.Linear(128, 64)
        self.bnp_5 = nn.BatchNorm1d(64)
        self.p_output_6 = nn.Linear(64, 16)
        self.bnp_6 = nn.BatchNorm1d(16)
        self.p_output_7 = nn.Linear(16, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        i_values = nn.ReLU()(self.bni_1(self.i_output_1(pooled_output)))
        i_values = nn.ReLU()(self.bni_2(self.i_output_2(i_values)))
        i_values = nn.ReLU()(self.bni_3(self.i_output_3(i_values)))
        i_values = nn.ReLU()(self.bni_4(self.i_output_4(i_values)))
        i_values = nn.ReLU()(self.bni_5(self.i_output_5(i_values)))
        i_values = nn.ReLU()(self.bni_6(self.i_output_6(i_values)))
        i_output = nn.Sigmoid()(self.i_output_7(i_values))
        
        n_values = nn.ReLU()(self.bnn_1(self.n_output_1(pooled_output)))
        n_values = nn.ReLU()(self.bnn_2(self.n_output_2(n_values)))
        n_values = nn.ReLU()(self.bnn_3(self.n_output_3(n_values)))
        n_values = nn.ReLU()(self.bnn_4(self.n_output_4(n_values)))
        n_values = nn.ReLU()(self.bnn_5(self.n_output_5(n_values)))
        n_values = nn.ReLU()(self.bnn_6(self.n_output_6(n_values)))
        n_output = nn.Sigmoid()(self.n_output_7(n_values))
        
        f_values = nn.ReLU()(self.bnf_1(self.f_output_1(pooled_output)))
        f_values = nn.ReLU()(self.bnf_2(self.f_output_2(f_values)))
        f_values = nn.ReLU()(self.bnf_3(self.f_output_3(f_values)))
        f_values = nn.ReLU()(self.bnf_4(self.f_output_4(f_values)))
        f_values = nn.ReLU()(self.bnf_5(self.f_output_5(f_values)))
        f_values = nn.ReLU()(self.bnf_6(self.f_output_6(f_values)))
        f_output = nn.Sigmoid()(self.f_output_7(f_values))

        p_values = nn.ReLU()(self.bnp_1(self.p_output_1(pooled_output)))
        p_values = nn.ReLU()(self.bnp_2(self.p_output_2(p_values)))
        p_values = nn.ReLU()(self.bnp_3(self.p_output_3(p_values)))
        p_values = nn.ReLU()(self.bnp_4(self.p_output_4(p_values)))
        p_values = nn.ReLU()(self.bnp_5(self.p_output_5(p_values)))
        p_values = nn.ReLU()(self.bnp_6(self.p_output_6(p_values)))
        p_output = nn.Sigmoid()(self.p_output_7(p_values))
        return [i_output, n_output, f_output, p_output]

# In[210]:


import time
from torch import nn, optim
import math

# In[211]:

# to calculate validation mse
def calculate_mse(model, loader):
    model.eval()
    mse1, mse2, mse3, mse4 = 0, 0, 0, 0
    mse = [mse1, mse2, mse3, mse4]
    with torch.no_grad():
        for X, mask, y in loader:
            y_pred = model(X, mask)
            for i in range(4):
                mse[i] += torch.mean((y_pred[i] - y[:, i]) ** 2).item() * len(X)
    model.train()
    mse = [single / len(loader) for single in mse]
    return sum(mse) / 4

# In[238]:


mbti_dict = {"i": "e", "n": "s", "f": "t", "p": "j"}
base = np.array(list(mbti_dict.keys()))
def mbti_val(scores, channel):
    base_index = list(mbti_dict.keys())[channel]
    res = []
    # cate_idx = np.argmax(scores, axis = 1)
    for score in scores:
        if score < 0.5:
            res.append(mbti_dict[base_index])
        else:
            res.append(base_index)
    return np.array(res)

# In[239]:


def calculate_accuracy(model, loader):
    model.eval()
    correct1, correct2, correct3, correct4 = 0, 0, 0, 0
    correct = [correct1, correct2, correct3, correct4]
    total = 0
    with torch.no_grad():
        for X, mask, y in loader:
            y_pred = model(X, mask)
            for i in range(4):
                pred = mbti_val(y_pred[i].cpu().detach().numpy(), i)
                target = mbti_val(y[:, i].cpu().numpy(), i)
                correct[i] += np.sum(pred == target)
            total += y.size(0)
    model.train()
    return [c / total * 100 for c in correct]

def calculate_f1(model, loader):
    model.eval()
    tp1, tp2, tp3, tp4 = 0, 0, 0, 0
    fp1, fp2, fp3, fp4 = 0, 0, 0, 0
    fn1, fn2, fn3, fn4 = 0, 0, 0, 0
    tp = [tp1, tp2, tp3, tp4]
    fp = [fp1, fp2, fp3, fp4]
    fn = [fn1, fn2, fn3, fn4]
    with torch.no_grad():
        for X, mask, y in loader:
            y_pred = model(X, mask)
            for i in range(4):
                pred = mbti_val(y_pred[i].cpu().detach().numpy(), i)
                target = mbti_val(y[:, i].cpu().numpy(), i)
                tp[i] += np.sum((pred == target) & (pred == base[i]))
                fp[i] += np.sum((pred != target) & (pred == base[i]))
                fn[i] += np.sum((pred != target) & (target == base[i]))
    model.train()
    f1 = []
    for i in range(4):
        f1_score = 2 * tp[i] / (2 * tp[i] + fp[i] + fn[i])
        f1.append(f1_score)
    return f1

# In[240]:


from datetime import datetime
def train_model(model, loss_func, num_epochs, optimizer, train_loader, val_loader):
    train_loss_log, val_acc_log = [], []
    tic = time.time()
    for epoch in range(num_epochs):
        train_loss = 0
        best_val_loss = 1e5
        for i, data in enumerate(train_loader):
            X, mask, y = data
            pred_y = model(X, mask)
            # for j in range(4):
            j = i % 4
            pred = pred_y[j]
            loss = loss_func(pred, y[:, j])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X)
        train_loss /= len(train_loader)
        train_loss_log.append(train_loss)
        val_acc = calculate_f1(model, val_loader)
        val_acc_mean = sum(val_acc) / 4
        val_acc_log.append(val_acc_mean)
        val_loss = calculate_mse(model, val_loader)
        print("Epoch {:2},  Training Loss: {:9.6f}, Validation Loss: {:9.6f}, Validation F1-i: {:7.6f}, n: {:7.6f}, f: {:7.6f}, p: {:7.6f}".format(epoch, train_loss, 
                                                                                                                        val_loss,
                                                                                                                          val_acc[0], 
                                                                                                                          val_acc[1],
                                                                                                                          val_acc[2], 
                                                                                                                          val_acc[3]))
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "variables/%s_model.pth"%train_time_token)
    toc = time.time()
    print(f"Training time: {toc - tic:.2f}s")
    return train_loss_log, val_acc_log

# In[241]:


model = mbti_bert(bert_model).to(device)
loss_func = nn.MSELoss()
# loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-3)
num_epochs = 400
train_time_token = datetime.now().strftime("%Y%m%d%H%M%S")
print(f"Training start at {train_time_token}")
print("seed is", seed)
train_loss_log, val_acc_log = train_model(model, loss_func, num_epochs, optimizer, train_loader, val_loader)
# training logs can be found in results/2024042913105_log.pkl
# In[ ]:


import pickle
pickle.dump([train_loss_log, val_acc_log, val_loader, 
             X_val, y_val, mask_val,
             X_test, y_test, mask_test], open("results/%s_log.pkl"%train_time_token, "wb"))
optimal = torch.load("variables/%s_model.pth"%train_time_token, map_location = device)
model.load_state_dict(optimal)
