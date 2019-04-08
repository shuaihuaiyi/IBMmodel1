import torch


class IBMModel1:
  def __init__(self, src_vocab, tgt_vocab, use_gpu=False):
    self.src_vocab = src_vocab
    self.tgt_vocab = tgt_vocab
    if use_gpu and torch.cuda.is_available():
      # self.device = torch.device('cuda')
      torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
      # self.device = torch.device('cpu')
      torch.set_default_tensor_type(torch.DoubleTensor)
    
    self.t = torch.full((len(src_vocab), len(tgt_vocab)), 1 / len(tgt_vocab))
  
  def step(self, data):
    count = torch.zeros_like(self.t)
    total = torch.zeros(len(self.src_vocab))
    s_total = torch.zeros(len(self.tgt_vocab))
    
    for sent_pair in data:
      for tgt_word in sent_pair[1]:
        s_total[tgt_word] = 0
        for src_word in sent_pair[0]:
          s_total[tgt_word] += self.t[src_word][tgt_word]
      for tgt_word in sent_pair[1]:
        for src_word in sent_pair[0]:
          count[src_word][tgt_word] += self.t[src_word][tgt_word] / s_total[tgt_word]
          total[src_word] += self.t[src_word][tgt_word] / s_total[tgt_word]
    
    for src_word in self.src_vocab:
      for tgt_word in self.tgt_vocab:
        self.t[src_word][tgt_word] = count[src_word][tgt_word] / total[src_word]
