from __future__ import print_function
import argparse
import json
import math
import time

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

from models import RNNModel

torch.backends.cudnn.enabled = False


def package(bsz):
  batch = list()
  for i in range(bsz):
    line = fin.readline()
    if not line:
      break
    batch.append(["<S>"] + json.loads(line)[0].split())
  lengths = [len(sent) for sent in batch]
  labels = [sent[1:] + ["</S>"] for sent in batch]
  return batch, lengths, labels


def word2id(word):
  if word in model.vocab:
    return model.vocab[word]
  else:
    return model.vocab["<UNK>"]


def batch2list(batch, lengths, labels):
  batch = [[word2id(word) for word in sent] for sent in batch]
  labels = [[word2id(word) for word in sent] for sent in labels]
  ids = [(i, lengths[i]) for i in range(len(lengths))]
  ids = sorted(ids, key=lambda x: -x[1])
  batch = [batch[idx[0]] for idx in ids]
  labels = [labels[idx[0]] for idx in ids]
  lengths = [lengths[idx[0]] for idx in ids]
  for i in range(len(batch)):
    for j in range(lengths[0] - len(batch[i])):
      batch[i].append(model.vocab["</S>"])
      labels[i].append(model.vocab["</S>"])
  return batch, lengths, labels


def valid():
  model.eval()
  total_loss = 0
  batch_id = 0
  for i in range(options.log_interval):
    batch_id += 1
    batch_data, batch_length, batch_label = package(options.batch_size)
    if len(batch_data) != options.batch_size:
      break
    batch_data, batch_length, batch_label = batch2list(batch_data, batch_length, batch_label)
    batch_data = Variable(torch.LongTensor(batch_data), volatile=True)
    batch_label = Variable(torch.LongTensor(batch_label), volatile=True)
    if options.cuda:
      batch_label = batch_label.cuda()
      batch_data = batch_data.cuda()
    batch_label = nn.utils.rnn.pack_padded_sequence(batch_label, batch_length, batch_first=True)

    output, hid = model(batch_data, batch_length)

    loss = criterion(output, batch_label.data)
    total_loss += loss.data[0]

  return total_loss / batch_id


def train(ep):
  global last_valid_loss
  batch_id = 0
  total_loss = 0
  start_time = time.time()
  model.train()
  end_of_epoch = False
  while not end_of_epoch:
    batch_data, batch_length, batch_label = package(options.batch_size)
    if len(batch_data) != options.batch_size:
      end_of_epoch = True
      continue
    batch_data, batch_length, batch_label = batch2list(batch_data, batch_length, batch_label)
    batch_data = Variable(torch.LongTensor(batch_data), requires_grad=False)
    batch_label = Variable(torch.LongTensor(batch_label), requires_grad=False)
    if options.cuda:
      batch_label = batch_label.cuda()
      batch_data = batch_data.cuda()
    batch_label = nn.utils.rnn.pack_padded_sequence(batch_label, batch_length, batch_first=True)

    output, hid = model(batch_data, batch_length)

    loss = criterion(output, batch_label.data)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), options.lstm_clip_norm)
    optimizer.step()

    # update metadata
    batch_id += 1
    total_loss += loss.data[0]
    if batch_id % options.log_interval == 0:
      loss = total_loss / options.log_interval
      elapsed = time.time() - start_time
      print("| epoch {:3d} | {:5d} batches | ms/batch {:6.2f} | loss {:5.2f} | ppl {:8.2f}".format(
        ep, batch_id, elapsed * 1000 / options.log_interval, loss, math.exp(loss)))
      start_time = time.time()
      total_loss = 0

    if batch_id % options.evaluation_interval == 0:
      loss = valid()
      print("-"*70)
      print("| validation | time {:6.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}".format(
        time.time() - start_time, loss, math.exp(loss)))
      print("learning rates:")
      for param_group in optimizer.param_groups:
        print(param_group["lr"])
      print("-"*70)
      if loss > last_valid_loss:
        for param_group in optimizer.param_groups:
          param_group["lr"] = param_group["lr"] / 4.0
      last_valid_loss = loss
      model.train()

      # save
      with open(options.model_prefix + '.%d-%d.pt' % (ep, batch_id), 'wb') as fmodel:
        torch.save(model, fmodel)
        fmodel.close()

      start_time = time.time()


"""
Training Commands:
CUDA_VISIBLE_DEVICES=2 python train.py -pre /misc/vlgscratch4/BowmanGroup/haoyue/models/glove.840B.300d.pt -lr 5 -data /misc/vlgscratch4/BowmanGroup/haoyue/json/GutenbergReal.json -prefix /misc/vlgscratch4/BowmanGroup/haoyue/models/generator_gutenberg_0 -eval 200
"""

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="RNNLM generator")

  # word embeddings
  parser.add_argument("-pre", "--pretrained_model", type=str, default="",
                      help="pre-trained word embeddings")
  parser.add_argument("-crop", "--dictionary_crop", type=int, default=100000,
                      help="keep most frequent words in the vocab")
  parser.add_argument("-emsize", "--embedding_size", type=int, default=300,
                      help="dimension of word embedding")
  parser.add_argument("-vocab", "--vocabulary", type=str, default="../Data/SortedVocab.txt",
                      help="external vocabulary")

  # LSTM generator
  parser.add_argument("-clip", "--lstm_clip_norm", type=float, default=5.0,
                      help="clip for LSTM to avoid too large grad")
  parser.add_argument("-nhid", "--hidden_size", type=int, default=512,
                      help="hidden units in the model")
  parser.add_argument("-nlayers", "--layer_num", type=int, default=2,
                      help="layer number in LSTM cell")

  # general model
  parser.add_argument("-drop", "--dropout", type=float, default=0.5,
                      help="dropout ratio for model training")
  parser.add_argument("-ncuda", "--no_cuda", action="store_false", dest="cuda",
                      help="not use CUDA")
  parser.add_argument("-epoch", "--epoch_number", type=int, default=10,
                      help="epoch number for training")
  parser.add_argument("-log", "--log_interval", type=int, default=200,
                      help="report log interval")
  parser.add_argument("-eval", "--evaluation_interval", type=int, default=50000,
                      help="evaluation interval")
  parser.add_argument("-bsz", "--batch_size", type=int, default=32,
                      help="batch size for training")

  # optimizer
  parser.add_argument("-lr", "--learning_rate", type=float, default=0.1,
                      help="learning rate")
  parser.add_argument("-optim", "--optimizer", type=str, default="sgd",
                      help="optimization algorithm [adagrad|adam|sgd]")

  # data and saving
  parser.add_argument("-data", "--data_path", type=str, default="",
                      help="path to the training data")
  parser.add_argument("-prefix", "--model_prefix", type=str, default="",
                      help="prefix for saving the model")

  options = parser.parse_args()
  print(options)

  # start training
  model = RNNModel(options)
  if options.cuda:
    model = model.cuda()

  criterion = nn.CrossEntropyLoss()
  if options.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=options.learning_rate)
  elif options.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=options.learning_rate)
  elif options.optimizer == "adagrad":
    optimizer = optim.Adagrad(model.parameters(), lr=options.learning_rate)
  else:
    raise Exception("optimizer %s not supported" % options.optimizer)

  try:
    last_valid_loss = 1e10
    for epoch in range(options.epoch_number):
      fin = open(options.data_path, "r")
      train(epoch)
      fin.close()
  except KeyboardInterrupt:
    print("Exit from training early.")
