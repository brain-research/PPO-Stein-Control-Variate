# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 121 --unbiased --state-only --dir-name=halfcheetah_unbiased--state-only_121 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 121 --unbiased --state-only --dir-name=humanoid_unbiased--state-only_121 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 144 --unbiased --state-only --dir-name=halfcheetah_unbiased--state-only_144 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 144 --unbiased --state-only --dir-name=humanoid_unbiased--state-only_144 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 169 --unbiased --state-only --dir-name=halfcheetah_unbiased--state-only_169 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 169 --unbiased --state-only --dir-name=humanoid_unbiased--state-only_169 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 225 --unbiased --state-only --dir-name=halfcheetah_unbiased--state-only_225 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 225 --unbiased --state-only --dir-name=humanoid_unbiased--state-only_225 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 256 --unbiased --state-only --dir-name=halfcheetah_unbiased--state-only_256 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 256 --unbiased --state-only --dir-name=humanoid_unbiased--state-only_256 &
