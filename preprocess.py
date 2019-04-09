import argparse
import pickle

from tqdm import tqdm


def read_corpus(src, tgt):
  result = []
  with open(src) as s, open(tgt) as t:
    for ss, ts in tqdm(zip(s, t), desc='Reading corpus '):
      result.append([ss.split(), ts.split()])
  return result


def build_vocab(corpus):
  vocab_tgt = {}
  i_tgt = 0
  vocab_src = {}
  i_src = 0
  for src_sent, tgt_sent in tqdm(corpus, desc='Building vocab '):
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
  for ss, ts in tqdm(corpus, desc='Converting corpus '):
    data.append([[], []])
    for sw in ss:
      data[-1][0].append(vocab_src[sw])
    for tw in ts:
      data[-1][1].append(vocab_tgt[tw])
  return data


def save_data(file_name, vocab_src, vocab_tgt, train_data):
  with open(file_name, 'wb') as f:
    pickle.dump((vocab_src, vocab_tgt, train_data), f)


def load_data(file_name):
  with open(file_name, 'rb') as f:
    return pickle.load(f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='IBM Model 1 preprocess module.')
  parser.add_argument('train_src')
  parser.add_argument('train_tgt')
  parser.add_argument('out_file')
  args = parser.parse_args()
  
  corpus = read_corpus(args.train_src, args.train_tgt)
  vocab_src, vocab_tgt = build_vocab(corpus)
  train_data = convert_corpus(corpus, vocab_src, vocab_tgt)
  
  save_data(args.out_file, vocab_src, vocab_tgt, train_data)
