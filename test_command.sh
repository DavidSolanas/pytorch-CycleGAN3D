#!/bin/bash
python test.py --dataroot /Disco2021-I/david/tfm/dataset/cyclegan_dataset/ --name Train500VolumesV2 --model cycle_gan3d --input_nc 1 --output_nc 1 --gpu_ids 1 --eval --load_iter 200