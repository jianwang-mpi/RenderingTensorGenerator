#!/usr/bin/env bash
input_path=/unsullied/sharefs/wangjian02/isilon-home/datasets/Market1501/data
output_path=/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market1501_rendering_matrix_new
python generate_matrix.py ${input_path} ${output_path}