import argparse

from tqdm import tqdm

from model import IBMModel1
import pickle

def read_corpus(src, tgt):
  result = []
  with open(src) as s, open(tgt) as t:
    for ss, ts in zip(s, t):
      result.append([ss.split(), ts.split()])
  return result


def build_vocab(corpus):
  vocab_tgt = {}
  i_tgt = 0
  vocab_src = {}
  i_src = 0
  for src_sent, tgt_sent in corpus:
    for src_word in src_sent:
      if src_word not in vocab_src:
        vocab_src[src_word] = i_src
        i_src += 1
    for tgt_word in tgt_sent:
      if tgt_word not in vocab_tgt:
        vocab_tgt[tgt_word] = i_tgt
        i_tgt += 1
  return vocab_src, vocab_tgt


def convert_corpus(corpus, vocab_src, vocab_tgt):
  data = []
  for ss, ts in corpus:
    data.append([[], []])
    for sw in ss:
      data[-1][0].append(vocab_src[sw])
    for tw in ts:
      data[-1][1].append(vocab_tgt[tw])
  return data


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='IBM Model 1 in PyTorch')
  parser.add_argument('train_src')
  parser.add_argument('train_tgt')
  parser.add_argument('out_file')
  parser.add_argument('--use_gpu', default=False, action='store_true')
  args = parser.parse_args()
  
  # todo 将数据预处理放入preprocess.py，并提供main入口
  corpus = read_corpus(args.train_src, args.train_tgt)
  vocab_src, vocab_tgt = build_vocab(corpus)
  train_data = convert_corpus(corpus, vocab_src, vocab_tgt)
  
  model = IBMModel1(vocab_src, vocab_tgt)
  # todo 编写合适的训练步骤
  for i in tqdm(range(1000)):
    model.step(train_data)
  
  # todo 实现正确的数据持久化方法
  with open('save.pkl', 'wb') as f:
    pickle.dump({'vocab_src':vocab_src, 'vocab_tgt':vocab_tgt, 't':model.t})
    