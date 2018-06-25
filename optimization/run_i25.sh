# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 121 --unbiased --dir-name=halfcheetah_unbiased_121 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 121 --unbiased --dir-name=humanoid_unbiased_121 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 144 --unbiased --dir-name=halfcheetah_unbiased_144 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 144 --unbiased --dir-name=humanoid_unbiased_144 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 169 --unbiased --dir-name=halfcheetah_unbiased_169 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 169 --unbiased --dir-name=humanoid_unbiased_169 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 225 --unbiased --dir-name=halfcheetah_unbiased_225 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 225 --unbiased --dir-name=humanoid_unbiased_225 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 256 --unbiased --dir-name=halfcheetah_unbiased_256 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 256 --unbiased --dir-name=humanoid_unbiased_256 &
