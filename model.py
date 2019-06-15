# -*- coding: utf-8 -*-
# @Time : 2019/6/1 下午4:19
# @Author : Sophie_Zhang
# @File : model.py
# @Software: PyCharm

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt, log

no_cuda = False

device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")


class BLSTM(nn.Module):
    """
        Implementation of BLSTM Concatenation for sentiment classification task
    """
    #hidden_dim=1024, output_dim=728
    def __init__(self, dim_model, hidden_dim, num_layers=1, dropout=0.5):
        super(BLSTM, self).__init__()
        self.input_dim = dim_model
        self.hidden_dim = hidden_dim
        # sen encoder
        self.sen_rnn = nn.LSTM(input_size=self.input_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        self.output = nn.Linear(2 * self.hidden_dim, self.input_dim)

    def forward(self, x):
        """
        :param sen_batch: (batch, sen_length), tensor for sentence sequence
        :return:
        """
        # print("check1")
        batch_size=x.size(0)
        # print("check2")

        ''' Bi-LSTM Computation '''
        # print(x)
        sen_outs, _ = self.sen_rnn(x.float())
        # print("check3")
        # print(type(sen_outs))

        representation = sen_outs.contiguous().view(batch_size, -1, 2 * self.hidden_dim)  # (batch, sen_len, 2*hid)
        out = self.output(representation)

        return out.float()

# #### Attention
# scaled dot product attention:
#  $\text { Attention }(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V$

def scaled_dot_product_attention(query, key, value, mask=None, dropout=0.5):
    dim_key = query.size(-1)
    # print(dim_key)
    # print(query.shape, key.transpose(-2, -1).shape)
    attn = torch.matmul(query, key.transpose(-2, -1)) / sqrt(dim_key)
    # print(mask.shape)
    mask = torch.matmul(mask, mask.transpose(-2, -1) / sqrt(mask.size(-1)))
    # print(attn)
    # print(torch.matmul(query, key.transpose(-2, -1)).shape)
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)
        # attn = attn.masked_fill(mask == 0, value=-np.inf)
    attn_weights = F.softmax(attn, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return torch.matmul(attn_weights, value), attn_weights

# Self attention: K=V=Q
# each word in the sentence needs to undergo Attention computation, to capture the internal structure of the sentence
#
# Multi-head Attention: query, key, and value first go through a linear transformation and then enters into Scaled-Dot Attention. Here, the attention is calculated h times, which allows the model to learn relevant information in different representative child spaces.
# When #head=1, it becomes a original self-attention layer.

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, dim_model, dropout=0.5):
        super(MultiHeadedAttention, self).__init__()
        # make sure input word embedding dimension divides by the number of desired heads
        assert dim_model % num_heads == 0
        # assume dim of key,query,values are equal
        self.dim_qkv = dim_model // num_heads

        self.dim_model = dim_model
        self.num_h = num_heads
        self.w_q = nn.Linear(dim_model, dim_model)  # self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_k = nn.Linear(dim_model, dim_model)
        self.w_v = nn.Linear(dim_model, dim_model)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # mask = mask.unsqueeze(1)
            mask = mask.repeat(self.num_h, 1, 1, 1)
            # print(mask.shape)

        n_batch = query.size(0)
        #         residual = query

        # linear projections: dim_model => num_h x dim_k
        query = self.w_q(query).view(n_batch, -1, self.num_h, self.dim_qkv).permute(2, 0, 1, 3)
        key = self.w_k(key).view(n_batch, -1, self.num_h, self.dim_qkv).permute(2, 0, 1, 3)
        value = self.w_v(value).view(n_batch, -1, self.num_h, self.dim_qkv).permute(2, 0, 1, 3)

        # query = self.w_q(query).view(n_batch, -1, self.num_h, self.dim_qkv)
        # key = self.w_k(key).view(n_batch, -1, self.num_h, self.dim_qkv)
        # value = self.w_v(value).view(n_batch, -1, self.num_h, self.dim_qkv)

        # query=query.permute(2,0,1,3)
        # query=query.view(n_batch*self.num_h,-1,self.dim_qkv)

        # print("query shape:{}".format(query.shape))

        # Apply attention on all the projected vectors in batch
        x, self.attn = scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.permute(1, 2, 0, 3)
        # print("shape0:{}".format(x.shape))

        # Concat(head1, ..., headh)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.num_h * self.dim_qkv)
        # print("shape:{}".format(x.shape))

        if torch.cuda.is_available():
            linear = nn.Linear(self.dim_model, self.dim_model, bias=False).to(device)
        else:
            linear = nn.Linear(self.dim_model, self.dim_model, bias=False)
        x = linear(x)
        #         x = self.layer_norm(x + residual)
        return x

# #### Position-wise feed forward network

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.5):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu # bert uses gelu instead
        # self.activation = F.glu # bert uses gelu instead

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


# #### Add & Norm
# `Residual connection`是对于较为深层的神经网络有比较好的作用，比如网络层很深时，数值的传播随着weight不断的减弱，`Residual connection`是从输入的部分，连到它输出层的部分，把输入的信息原封不动copy到输出的部分，减少信息的损失。
# `layer-normalization`这种归一化层是为了防止在某些层中由于某些位置过大或者过小导致数值过大或过小，对神经网络梯度回传时有训练的问题，保证训练的稳定性。基本在每个子网络后面都要加上`layer-normalization`、加上`Residual connection`，加上这两个部分能够使深层神经网络训练更加顺利。
# (本实验中也许不需要)

class AddNorm(nn.Module):
    def __init__(self, size, dropout, eps=1e-6):
        super(AddNorm, self).__init__()
        # self.a_2 = nn.Parameter(torch.ones(size))
        # self.b_2 = nn.Parameter(torch.zeros(size))
        # self.eps = eps
        self.NormLayer=nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x = x.float()
        # mean = x.mean(-1, keepdim=True)
        # std = x.std(-1, keepdim=True)
        # norm = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        norm = self.NormLayer(x)
        return x + self.dropout(sublayer(norm))


# #### Encoder
# self-attention layers: all of the keys, values and queries come from the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.

# 一层Encoder: self-atten --> add&norm --> feed-forward --> add&norm
# 浅层网络可以去掉add&norm层？
class EncoderLayer(nn.Module):
    def __init__(self, size, attention, feed_forward, dropout=0.5):
        super(EncoderLayer, self).__init__()
        self.feed_forward = feed_forward
        self.self_atten = attention
        self.add_norm_1 = AddNorm(size, dropout)
        self.add_norm_2 = AddNorm(size, dropout)
        self.size = size

    def forward(self, x, mask=None):
        output = self.add_norm_1(x, lambda x: self.self_atten(x, x, x, mask))
        output = self.add_norm_2(output, self.feed_forward)
        # output = self.add_norm_2(x, self.feed_forward)
        return output


class Encoder(nn.Module):
    def __init__(self, dim_model, layer, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)]) # clone the layer for N times
        self.norm = nn.LayerNorm(layer.size)
        self.dim_model = dim_model

    def forward(self, x, mask=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # x = self.positional_encoding(x, self.dim_model)

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    def positional_encoding(self, x, dim_model, max_len=5000):
        # whether dropout?
        sentence_len = x.size(1)
        pe_vec = torch.zeros(max_len, dim_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., dim_model, 2) * -(log(10000.0) / dim_model))
        # pe_vec[:, 0::2] = torch.sin(position)
        # pe_vec[:, 1::2] = torch.cos(position)
        pe_vec[:, 0::2] = torch.sin(position * div_term)
        pe_vec[:, 1::2] = torch.cos(position * div_term)
        pe_vec = pe_vec.unsqueeze(0)
        # print(x.shape, pe_vec.shape)
        return x.float() + pe_vec[:, :sentence_len]
        # return pe_vec[:, :sentence_len]

# #### Classifier

class SoftMax(nn.Module):
    def __init__(self, n_input, n_out):
        super(SoftMax, self).__init__()
        self.fc = nn.Linear(n_input, n_out)
        self.softmax = nn.LogSoftmax(1)
        # self.softmax = nn.Softmax()

    def forward(self, x, add_feature):
        # print("xxxx shape:{}".format(x.shape))
        # print("feature shape:{}".format(add_feature.shape))
        # print(x.shape)
        # print(torch.from_numpy(add_feature).shape)
        # print(x)
        # x_feats = torch.cat((x, torch.from_numpy(add_feature)), dim=1)
        x_feats = torch.cat((x.float(), add_feature.float()), dim=1)
        # print(x_feats.shape)
        # x = F.relu(self.fc(x_feats))
        x = self.fc(x_feats)
        y = self.softmax(x)
        #         print(y)
        # print(y)
        return y


# #### Full Model

# single task
# embeding --> encoder --> linear --> softmax
class SelfAttenClassifier(nn.Module):
    def __init__(self, bilstm, encoder, classifier):
        super(SelfAttenClassifier, self).__init__()
        self.bilstm = bilstm
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, input_embeds, add_feature, mask_tensor):
        # BiLSTM -> transformer -> classifier
        batch_size = input_embeds.size(1)
        # bilstm_out = self.bilstm(input_embeds)
        # encoder_out = self.encoder(bilstm_out, mask_tensor)
        # # Simply average the final sequence position representations to create a fixed size "sentence representation".
        # #         sentence_representation = tf.reduce_mean(encoder_output, axis=1)    # [batch_size, model_dim]
        # feats = encoder_out.sum(dim=1)
        # #         print(encoder_out.size(), feats.size())
        # outputs = self.classifier(feats, add_feature)
        # outputs = self.classifier(bilstm_out, add_feature)

        # transformer -> BiLSTM -> classifier
        encoder_out = self.encoder(input_embeds, mask_tensor)
        bilstm_out = self.bilstm(encoder_out)
        feats = bilstm_out.sum(dim=1)
        outputs = self.classifier(feats, add_feature)

        return outputs