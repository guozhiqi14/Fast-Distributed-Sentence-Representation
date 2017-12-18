from __future__ import print_function

import argparse
import shelve

import TextItem

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Pre-processing for text files.")
  parser.add_argument("--input", type=str, default="",
                      help="input shelve file")
  parser.add_argument("--output", type=str, default="",
                      help="output text file")
  options = parser.parse_args()

  input_db = shelve.open(options.input)
  output_db = shelve.open(options.output)
  input_idx = output_idx = 0
  valid_batch = train_batch = 0
  batch_size = 64000
  if options.find("Wikipedia") != -1:
    sample_range = 1000
  else:
    sample_range = 2000

  curr_valid = list()
  curr_train = list()

  while str(input_idx) in input_db:
    for para in input_db[str(input_idx)].paragraphs:
      for sent in para:
        sent = sent.lower().split()
        output_idx += 1
        if output_idx % sample_range == 0:
          curr_valid.append(sent)
          if len(curr_valid) == batch_size:
            output_db["valid-%d" % valid_batch] = curr_valid
            valid_batch += 1
            curr_valid = list()
        else:
          curr_train.append(sent)
          if len(curr_train) == batch_size:
            output_db["train-%d" % train_batch] = curr_train
            train_batch += 1
            curr_train = list()
    input_idx += 1
    if input_idx % 1000 == 0:
      print(input_idx, output_idx, valid_batch, train_batch, len(curr_valid))
      output_db.sync()

  output_db["valid-%d" % valid_batch] = curr_valid
  output_db["train-%d" % train_batch] = curr_train

  output_db.sync()
  input_db.close()
  output_db.close()
