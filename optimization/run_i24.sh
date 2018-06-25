# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 121 --extra-sample --dir-name=halfcheetah_extra-sample_121 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 121 --extra-sample --dir-name=humanoid_extra-sample_121 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 144 --extra-sample --dir-name=halfcheetah_extra-sample_144 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 144 --extra-sample --dir-name=humanoid_extra-sample_144 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 169 --extra-sample --dir-name=halfcheetah_extra-sample_169 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 169 --extra-sample --dir-name=humanoid_extra-sample_169 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 225 --extra-sample --dir-name=halfcheetah_extra-sample_225 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 225 --extra-sample --dir-name=humanoid_extra-sample_225 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 256 --extra-sample --dir-name=halfcheetah_extra-sample_256 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 256 --extra-sample --dir-name=humanoid_extra-sample_256 &
