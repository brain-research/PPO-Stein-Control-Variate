# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

python train.py Humanoid-v1 -b 10000 -ps small -c 0 -s 343 --dir-name=humanoid_baseline_343 &
python train.py HalfCheetah-v1 -b 10000 -ps large -c 0 -s 454 --dir-name=halfcheetah_baseline_454 &
python train.py Humanoid-v1 -b 10000 -ps small -c 0 -s 454 --dir-name=humanoid_baseline_454 &
python train.py HalfCheetah-v1 -b 10000 -ps large -c 0 -s 565 --dir-name=halfcheetah_baseline_565 &
python train.py Humanoid-v1 -b 10000 -ps small -c 0 -s 565 --dir-name=humanoid_baseline_565 &

# python train.py Walker2d-v1 -b 10000 -ps large -c 0 -s 121 --dir-name=walker2d_baseline_121 &
# python train.py Walker2d-v1 -b 10000 -ps large -c 0 -s 144 --dir-name=walker2d_baseline_144 &
# python train.py Walker2d-v1 -b 10000 -ps large -c 0 -s 169 --dir-name=walker2d_baseline_169 &
# python train.py Walker2d-v1 -b 10000 -ps large -c 0 -s 196 --dir-name=walker2d_baseline_196 &
# python train.py Walker2d-v1 -b 10000 -ps large -c 0 -s 225 --dir-name=walker2d_baseline_225 &
