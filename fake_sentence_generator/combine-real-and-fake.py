from __future__ import print_function

import argparse
import json
import shelve


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Real and fake sentences combiner")
  parser.add_argument("--real", type=str, default="", help="path to real sentences")
  parser.add_argument("--fake", type=str, default="", help="path to fake sentences")
  parser.add_argument("--output", type=str, default="", help="path to output shelve")
  options = parser.parse_args()

  f_real = open(options.real)
  f_fake = open(options.fake)
  db = shelve.open(options.output)
  curr_batch = list()
  db_length = 0

  while True:
    l_real = f_real.readline()
    l_fake = f_fake.readline()
    if (not l_real) or (not l_fake):
      break
    curr_batch.append(json.loads(l_real))
    curr_batch.append(json.loads(l_fake))
    if len(curr_batch) == 64000:
      db[str(db_length)] = curr_batch
      db_length += 1
      curr_batch = list()
      print(db_length)
      db.sync()

  db[str(db_length)] = curr_batch
  db_length += 1
  db["len"] = db_length

  db.sync()
  db.close()
  f_real.close()
  f_fake.close()