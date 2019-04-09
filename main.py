import argparse
import csv

import numpy as np
from tqdm import tqdm

from model import IBMModel1
from preprocess import load_data


def write_to_file(out_file, vocab_src, vocab_tgt, t):
  # 交换索引项
  inv_vocab_src = {}
  inv_vocab_tgt = {}
  for item in vocab_src:
    inv_vocab_src[vocab_src[item]] = item
  for item in vocab_tgt:
    inv_vocab_tgt[vocab_tgt[item]] = item
  
  # 处理为列表
  temp = []
  for src_wordid, tgt_words in enumerate(t):
    for tgt_wordid, align_prob in enumerate(tgt_words):
      if align_prob > 1e-4:
        temp.append((inv_vocab_src[src_wordid], inv_vocab_tgt[tgt_wordid], align_prob))
  temp = sorted(temp, key=lambda x: x[2], reverse=True)
  with open(out_file, 'w') as f:
    w = csv.writer(f)
    for item in temp:
      w.writerow(item)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='IBM Model 1')
  parser.add_argument('train_data', help='Path to the pre-processed training data, should be a python pickle dump file')
  parser.add_argument('out_file', help='Path to the file used to save the alignment score')
  parser.add_argument('--max_step', help='Maximum iteration steps', default=100, type=int)
  parser.add_argument('--epsilon', help='Converge threshold', default=1e-4, type=float)
  args = parser.parse_args()
  
  vocab_src, vocab_tgt, train_data = load_data(args.train_data)
  
  model = IBMModel1(vocab_src, vocab_tgt)
  old_t = model.t
  for i in tqdm(range(args.max_step)):
    model.step(train_data)
    if np.linalg.norm(model.t - old_t) < args.epsilon:
      break
  
  write_to_file(args.out_file, vocab_src, vocab_tgt, model.t)
