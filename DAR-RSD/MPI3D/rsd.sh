#!/usr/bin/env bash
python train_rsd.py --gpu_id 0 --src rc --tgt rl --tradeoff 0.001 --tradeoff2 0.03 | tee ./rc2rl_0001_003.log
python train_rsd.py --gpu_id 0 --src rl --tgt rc --tradeoff 0.001 --tradeoff2 0.03 | tee ./rl2rc_0001_003.log
python train_rsd.py --gpu_id 0 --src rl --tgt t --tradeoff 0.001 --tradeoff2 0.03 | tee ./rl2rc_0001_003.log
python train_rsd.py --gpu_id 0 --src rc --tgt t --tradeoff 0.003 --tradeoff2 0.03 | tee ./rc2t_0003_003.log
python train_rsd.py --gpu_id 0 --src t --tgt rc --tradeoff 0.003 --tradeoff2 0.03 | tee ./t2rc_0003_003.log
python train_rsd.py --gpu_id 0 --src t --tgt rl --tradeoff 0.003 --tradeoff2 0.03 | tee ./t2rl_0003_003.log