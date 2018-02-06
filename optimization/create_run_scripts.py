import os
import math
import numpy as np

def main():
  for i, extra_option in enumerate(['', '--extra-sample', '--unbiased', '--unbiased --state-only']):
    fname = "run_i{}.sh".format(i+23)
    with open(fname, 'w+') as f:
      for seed in [121, 144, 169, 225, 256]:
        line = "python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s {} {} &\n".format(seed, extra_option)
        f.write(line)
        line = "python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s {} {}&\n".format(seed, extra_option)
        f.write(line)

if __name__ == '__main__':
  main()
