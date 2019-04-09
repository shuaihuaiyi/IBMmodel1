import numpy as np


class IBMModel1:
  def __init__(self, src_vocab, tgt_vocab):
    self.src_vocab = src_vocab
    self.tgt_vocab = tgt_vocab

    self.t = np.full((len(src_vocab), len(tgt_vocab)), 1 / len(tgt_vocab))
  
  def step(self, data):
    # M-step
    # todo 优化M步的执行效率
    count = np.zeros_like(self.t)
    total = np.zeros(len(self.src_vocab))
    s_total = np.zeros(len(self.tgt_vocab))
    
    for sent_pair in data:
      for tgt_word in sent_pair[1]:
        s_total[tgt_word] = 0
        for src_word in sent_pair[0]:
          s_total[tgt_word] += self.t[src_word][tgt_word]
      for tgt_word in sent_pair[1]:
        for src_word in sent_pair[0]:
          count[src_word][tgt_word] += self.t[src_word][tgt_word] / s_total[tgt_word]
          total[src_word] += self.t[src_word][tgt_word] / s_total[tgt_word]
    
    # E-step
    # for src_word in self.src_vocab:
    #   for tgt_word in self.tgt_vocab:
    #     self.t[src_word][tgt_word] = count[src_word][tgt_word] / total[src_word]
    self.t = (count.transpose(1, 0) / total).transpose(1, 0)
