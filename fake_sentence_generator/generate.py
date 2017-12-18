from __future__ import print_function

import argparse
import json

import torch
from torch.autograd import Variable

# wikipedia: 38455401
# gutenberg: 59176904

if __name__ == "__main__":
  parser = argparse.ArgumentParser("fake sentence generator")
  parser.add_argument("--model", type=str, default="",
                      help="model for generation")
  parser.add_argument("--output", type=str, default="",
                      help="output file")
  parser.add_argument("--temperature", type=float, default=1,
                      help="temperature for generation")
  parser.add_argument("--sentences", type=int, default=1000,
                      help="generate number")
  parser.add_argument("--no-cuda", action="store_false", dest="cuda",
                      help="not use CUDA")
  parser.add_argument("--batch-size", type=int, default=3200,
                      help="batch size for generation")
  parser.add_argument("--sentence_len", type=int, default=100,
                      help="batch size for generation")

  options = parser.parse_args()

  # load model
  model = torch.load(options.model)
  if options.cuda:
    model = model.cuda()
  model.eval()
  model.options.batch_size = options.batch_size

  # prepare vocab
  vocab = [""] * len(model.vocab)
  for word in model.vocab:
    vocab[model.vocab[word]] = word
  print(len(model.vocab))

  # generate
  with open(options.output, "w") as fout:
    generated_sentences = 0
    inp = Variable(torch.LongTensor(
      [[model.vocab["<S>"]] for i in range(options.batch_size)]
    ), volatile=True)
    lengths = [1] * options.batch_size
    if options.cuda:
      inp.data = inp.data.cuda()
    while generated_sentences < options.sentences:
      sentences = [["<S>"] for i in range(options.batch_size)]
      hid = None
      for i in range(options.sentence_len):
        output, hid = model(inp, lengths, hid)
        word_weights = output.squeeze().data.div(options.temperature).exp()
        word_idx = torch.multinomial(word_weights, 1)
        inp = Variable(word_idx, volatile=True).view(-1, 1)
        for j in range(options.batch_size):
          sentences[j].append(vocab[word_idx[j][0]])

      for i in range(options.batch_size):
        pos = -1
        for j in range(options.sentence_len):
          if sentences[i][j] == "</S>":
            pos = j
            break
        if pos < 3:
          continue
        generated_sentences += 1
        fout.write(json.dumps((" ".join(sentences[i][1:pos]), 0)) + "\n")
      print('| Generated {} sentences'.format(generated_sentences))
    fout.close()
