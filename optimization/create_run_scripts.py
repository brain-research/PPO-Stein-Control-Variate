# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import math
import numpy as np

def main():
  for i, extra_option in enumerate(['', '--extra-sample', '--unbiased', '--unbiased --state-only']):
    fname = "run_i{}.sh".format(i+23)
    with open(fname, 'w+') as f:
      for seed in [121, 144, 169, 225, 256]:
        logdir = 'halfcheetah_{}_{}'.format(''.join(extra_option.split(" "))[2:], seed)
        line = "python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s {} {} --dir-name={} &\n".format(seed, extra_option, logdir)
        f.write(line)
        logdir = 'humanoid_{}_{}'.format(''.join(extra_option.split(" "))[2:], seed)
        line = "python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s {} {} --dir-name={} &\n".format(seed, extra_option, logdir)
        f.write(line)

if __name__ == '__main__':
  main()
