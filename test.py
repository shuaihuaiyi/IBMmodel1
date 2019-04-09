from model import IBMModel1


def compare_with_textbook(t):
  print(t[0][0], t[0][2], t[0][1], t[2][0], t[2][2], t[2][3], t[3][2], t[3][3], t[1][0], t[1][1])


if __name__ == '__main__':
  # src: das haus buch ein
  # tgt: the house book a
  #
  src_vocab = [0, 1, 2, 3]
  tgt_vocab = [0, 1, 2, 3]
  data = (((0, 1), (0, 1)), ((0, 2), (0, 2)), ((3, 2), (3, 2)))
  
  model = IBMModel1(src_vocab, tgt_vocab)
  
  while 1:
    compare_with_textbook(model.t)
    model.step(data)
