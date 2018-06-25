# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 121  --dir-name=halfcheetah__121 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 121  --dir-name=humanoid__121 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 144  --dir-name=halfcheetah__144 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 144  --dir-name=humanoid__144 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 169  --dir-name=halfcheetah__169 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 169  --dir-name=humanoid__169 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 225  --dir-name=halfcheetah__225 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 225  --dir-name=humanoid__225 &
python train.py HalfCheetah-v1 -b 10000 -ps large -po MinVar -p 500 -s 256  --dir-name=halfcheetah__256 &
python train.py Humanoid-v1 -b 10000 -ps small -po MinVar -p 500 -s 256  --dir-name=humanoid__256 &
