from __future__ import print_function

import argparse
import json
import shelve
import os


def multiple_q_mark(ws):
  cnt = 0
  for w in ws:
    if w == "?":
      cnt += 1
      if cnt > 1:
        return True
  return False


if __name__ == "__main__":
  parser = argparse.ArgumentParser("real sentences selector")
  parser.add_argument("--input", type=str, default="",
                      help="input shelve")
  parser.add_argument("--output", type=str, default="",
                      help="output json file")
  options = parser.parse_args()

  db = shelve.open(options.input)
  fout = open(options.output, "w")

  idx = 0
  sent_id = 0
  while db.has_key(str(idx)):
    for para in db[str(idx)].paragraphs:
      if len(para) > 8:
        for sent in para:
          words = sent.split()
          if len(words) > 100 or multiple_q_mark(words):
            continue
          fout.write(json.dumps((sent.lower(), 1)) + "\n")
          sent_id += 1
    idx += 1
    if idx % 1000 == 0:
      print(idx, sent_id)

  fout.close()
  db.close()

  os.system("shuf %s > %s.tmp" % (options.output, options.output))
  os.system("mv -f %s.tmp %s" % (options.output, options.output))
