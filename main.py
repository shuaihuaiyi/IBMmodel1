import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='IBM Model 1 in PyTorch')
  parser.add_argument('--use_gpu', default=False)
  parser.parse_args()