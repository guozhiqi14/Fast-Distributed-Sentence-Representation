from __future__ import print_function
import os

import torch
import torch.nn as nn
from torch.autograd import Variable


class RNNModel(nn.Module):
  def __init__(self, options):
    super(RNNModel, self).__init__()
    self.encoder = nn.Embedding(options.dictionary_crop + 2, options.embedding_size)
    self.rnn = nn.LSTM(options.embedding_size, options.hidden_size, options.layer_num, dropout=options.dropout)
    self.decoder = nn.Linear(options.hidden_size, options.dictionary_crop + 2)
    self.drop = nn.Dropout(options.dropout)
    self.options = options
    self.init_weights()
    if os.path.exists(options.pretrained_model):
      dictionary, embeddings, dim = torch.load(options.pretrained_model)
      keys = [word.strip() for word in open(options.vocabulary).readlines()[:options.dictionary_crop]] + ["<S>", "</S>"]
      self.vocab = dict([(keys[i], i) for i in range(len(keys))])
      embeds = map(lambda w: embeddings[dictionary[w]].view(1, -1) if w in dictionary else torch.zeros(1, dim), keys)
      self.encoder.weight.data = torch.cat(embeds, dim=0)

  def init_weights(self, init_range=0.1):
    self.decoder.bias.data.uniform_(-init_range, init_range)
    self.decoder.weight.data.uniform_(-init_range, init_range)

  def forward(self, inp, lengths, hid=None):
    emb = self.drop(self.encoder(inp))
    if not hid:
      hid = self.init_hidden()
    emb = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True)
    outp, hid = self.rnn(emb, hid)
    outp = self.drop(outp.data)
    decoded = self.decoder(outp.view(-1, outp.size(1)))
    return decoded.view(outp.size(0), -1), hid

  def init_hidden(self):
    weight = next(self.parameters()).data
    return (Variable(weight.new(self.options.layer_num, self.options.batch_size, self.options.hidden_size).zero_()),
            Variable(weight.new(self.options.layer_num, self.options.batch_size, self.options.hidden_size).zero_()))

  def __call__(self, inp, lengths, hid=None):
    return self.forward(inp, lengths, hid)
