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