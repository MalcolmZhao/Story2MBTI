#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings("ignore")

# In[3]:


import numpy as np
import pandas as pd
import random
# In[4]:

seed = 1307
raw_data = pd.read_excel("data/MVMR_BERT_tok.xlsx")

# In[5]:

# convert scores into the range [0, 1]
raw_data[["i_score", "n_score", "f_score", "p_score"]] = raw_data[["i_score", "n_score", "f_score", "p_score"]] / 100

# In[6]:
random.seed(seed)
np.random.seed(seed)


# In[7]:


import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# In[8]:

# create word index
tokenizer = get_tokenizer("basic_english")
def yield_tokens(data):
    for text in data:
        yield tokenizer(text)

# In[9]:


from collections import Counter
counter = Counter()
for tokens in yield_tokens(raw_data["dialog"]):
    counter.update(tokens)

# In[10]:


tokens = [token for token, freq in counter.items()]

# In[11]:


vocab = build_vocab_from_iterator(yield_tokens(raw_data["dialog"]), specials = ["<unk>", "<pad>", "<bos>", "<eos>"])
print(f"Vocab size: {len(vocab)}")

# In[12]:


def numericalize(data):
    for dialog in data:
        list = []
        for token in tokenizer(dialog):
            if token in vocab:
                list.append(vocab[token])
            else:
                list.append(vocab["<unk>"])
        yield list

# In[13]:


dialog_index = list(numericalize(raw_data["dialog"]))

# In[14]:


def text_pipeline(text):
    res = []
    for token in tokenizer(text):
        if token in vocab:
            res.append(vocab[token])
        else:
            res.append(vocab["<unk>"])
    return torch.tensor(res, dtype = torch.long)
def value_pipeline(target):
    return torch.tensor(target, dtype = torch.float32)

# In[15]:


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cuda detected")
else:
    device = torch.device("cpu")
    print("Running on CPU")

# In[16]:


X = [text_pipeline(dialog) for dialog in raw_data["dialog"]]
y = [value_pipeline(target) for target in raw_data[["i_score", "n_score", "f_score", "p_score"]].values.T]
from torch.nn.utils.rnn import pad_sequence
X = pad_sequence(X, batch_first = True, padding_value = vocab["<pad>"]).to(device)
y = torch.stack(y, dim = 1).to(device)

# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test_tmp, y_train, y_test_tmp = train_test_split(X, y, test_size = 0.2)
X_valid, X_test, y_valid, y_test = train_test_split(X_test_tmp, y_test_tmp, test_size = 0.5)
print(f"Train size: {len(X_train)}")
print(f"Valid size: {len(X_valid)}")
print(f"Test size: {len(X_test)}")

# In[18]:


from torch.utils.data import DataLoader, Dataset
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# In[19]:


train_dataset = TextDataset(X_train, y_train)
valid_dataset = TextDataset(X_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(valid_dataset, batch_size = 32, shuffle = True)

# In[20]:


import time

# In[21]:


def calculate_mse(loader, model):
    model.eval()
    mse1, mse2, mse3, mse4 = 0, 0, 0, 0
    mse = [mse1, mse2, mse3, mse4]
    with torch.no_grad():
        for X, y in loader:
            y_pred = model(X)
            for i in range(4):
                mse[i] += torch.mean((y_pred[:, i] - y[:, i]) ** 2).item()
    model.train()
    mse = [single / len(loader.dataset) for single in mse]
    return mse

# In[29]:

from datetime import datetime
def train_model(model, loss_func, num_epochs, optimizer, train_loader, val_loader):
    train_loss_log, val_loss_log = [], []
    tic = time.time()
    for epoch in range(num_epochs):
        train_loss = 0
        best_val_loss = 1
        for i, data in enumerate(train_loader):
            X, y = data
            pred_y = model(X)
            i = epoch % 4
            loss = loss_func(pred_y[:, i], y[:, i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X)
        train_loss /= len(train_loader)
        train_loss_log.append(train_loss)
        val_loss = calculate_mse(val_loader, model)
        val_loss_log.append(val_loss)
        print("Epoch {:2},  Training Loss: {:9.6f}, Validation Loss-i: {:7.6f}, n: {:7.6f}, f: {:7.6f}, p: {:7.6f}".format(epoch, train_loss, 
                                                                                                                          val_loss[0], 
                                                                                                                          val_loss[1],
                                                                                                                          val_loss[2], 
                                                                                                                          val_loss[3]))
        cur_val_loss = sum(val_loss)
        if cur_val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "variables/%s_model.pth"%train_time_token)
    toc = time.time()
    print(f"Training time: {toc - tic:.2f}s")
    return train_loss_log, val_loss_log


# In[23]:


from torch import nn, optim
import math

# In[1]:
# load word2vec embeddings
from gensim.models import KeyedVectors
print("Load word2vec embeddings")
word_embeddings = KeyedVectors.load_word2vec_format('word2vec.bin', binary=True)
print("Loading finished")

# In[24]:

embedding_weights = []
for token, idx in sorted(vocab.get_stoi().items(), key = lambda x: x[1]):
    if token in word_embeddings:
        embedding_weights.append(torch.tensor(word_embeddings[token]))
    else:
        embedding_weights.append(torch.zeros(word_embeddings.vectors[0].shape))

# In[25]:


embedding_weights = torch.stack(embedding_weights).to(device)


# In[31]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5300):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class mbti(nn.Module):
    def __init__(self, num_layers = 2, nhead = 3, dim_feedforward = 512, dropout = 0.1):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)
        self.pos_encoder = PositionalEncoding(300)
        encoder_layers = nn.TransformerEncoderLayer(d_model = 300, nhead = nhead, dim_feedforward = dim_feedforward, 
                                                    dropout = dropout, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers = num_layers)
        self.fc1 = nn.Linear(300, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 128)
        self.i_output_1 = nn.Linear(128, 64)
        self.i_output_2 = nn.Linear(64, 64)
        self.i_output_3 = nn.Linear(64, 32)
        self.i_output_4 = nn.Linear(32, 1)
        self.n_output_1 = nn.Linear(128, 64)
        self.n_output_2 = nn.Linear(64, 64)
        self.n_output_3 = nn.Linear(64, 32)
        self.n_output_4 = nn.Linear(32, 1)
        self.f_output_1 = nn.Linear(128, 64)
        self.f_output_2 = nn.Linear(64, 64)
        self.f_output_3 = nn.Linear(64, 32)
        self.f_output_4 = nn.Linear(32, 1)
        self.p_output_1 = nn.Linear(128, 64)
        self.p_output_2 = nn.Linear(64, 64)
        self.p_output_3 = nn.Linear(64, 32)
        self.p_output_4 = nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = self.pos_encoder(embedded)
        # embedded = [batch size, sent len, emb dim]
        out = self.transformer_encoder(embedded)
        # out = [batch size, sent len, emb dim]
        first_h = out[:, 0, :]
        out = nn.ReLU()(self.fc1(first_h))
        out = nn.ReLU()(self.fc2(out))
        out = nn.ReLU()(self.fc3(out))
        out = nn.ReLU()(self.fc4(out))
        i_out = nn.ReLU()(self.i_output_1(out))
        i_out = nn.ReLU()(self.i_output_2(i_out))
        i_out = nn.ReLU()(self.i_output_3(i_out))
        i_out = nn.Sigmoid()(self.i_output_4(i_out))
        n_out = nn.ReLU()(self.n_output_1(out))
        n_out = nn.ReLU()(self.n_output_2(n_out))
        n_out = nn.ReLU()(self.n_output_3(n_out))
        n_out = nn.Sigmoid()(self.n_output_4(n_out))
        f_out = nn.ReLU()(self.f_output_1(out))
        f_out = nn.ReLU()(self.f_output_2(f_out))
        f_out = nn.ReLU()(self.f_output_3(f_out))
        f_out = nn.Sigmoid()(self.f_output_4(f_out))
        p_out = nn.ReLU()(self.p_output_1(out))
        p_out = nn.ReLU()(self.p_output_2(p_out))
        p_out = nn.ReLU()(self.p_output_3(p_out))
        p_out = nn.Sigmoid()(self.p_output_4(p_out))
        return torch.cat([i_out, n_out, f_out, p_out], dim = 1)

# In[32]:


model = mbti().to(device)
model.embedding.weight.requires_grad = False # freeze embedding layer
loss_func = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-4)
num_epochs = 400
train_time_token = datetime.now().strftime("%Y%m%d%H%M%S")
print(f"Training start at {train_time_token}")
print("seed is", seed)
train_loss_log, val_loss_log = train_model(model, loss_func, num_epochs, optimizer, train_loader, val_loader)
# training logs can be found in results/20240422101530.pkl
# In[ ]:




import pickle
pickle.dump([train_loss_log, val_loss_log], open("results/%s_log.pkl"%train_time_token, "wb"))
optimal = torch.load("variables/%s_model.pth"%train_time_token, map_location = device)
model.load_state_dict(optimal)


mbti_dict = {"i": "e", "n": "s", "f": "t", "p": "j"}
def mbti_val(scores, channel):
    base_index = list(mbti_dict.keys())[channel]
    res = []
    for score in scores:
        if score < 0.5:
            res.append(mbti_dict[base_index])
        else:
            res.append(base_index)
    return np.array(res)

def calculate_accuracy(model, loader):
    model.eval()
    correct1, correct2, correct3, correct4 = 0, 0, 0, 0
    correct = [correct1, correct2, correct3, correct4]
    total = 0
    with torch.no_grad():
        for X, y in loader:
            y_pred = model(X)
            for i in range(4):
                pred = mbti_val(y_pred[:, i].cpu().numpy(), i)
                target = mbti_val(y[:, i].cpu().numpy(), i)
                correct[i] += np.sum(pred == target)
            total += y.size(0)
    model.train()
    return [c / total * 100 for c in correct]
print("validation accuracy:", calculate_accuracy(model, val_loader))
# results on val set around 54% for 4 channels

from sklearn.metrics import accuracy_score
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)
    y_test_pred = y_test_pred.cpu().numpy()
    y_test = y_test.cpu().numpy()
    y_test_label_pred = []
    y_test_label = []
    for i in range(4):
        y_test_label_pred.append(mbti_val(y_test_pred[:, i], i))
        y_test_label.append(mbti_val(y_test[:, i], i))
    y_test_label_pred = np.array(y_test_label_pred)
    y_test_label = np.array(y_test_label)
    acc = [accuracy_score(y_test_label[i], y_test_label_pred[i]) * 100 for i in range(4)]
print("test accuracy:", acc)
# results on test set around 56% for 4 channels