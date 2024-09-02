
conda activate dddp

######################## Please check your configurations first! ########################

#                   ID      Strict_ID   batch-size  train   valid   test    resume  tiny_dataset    gpu
python src/main.py Debug    false       5           true    true    true    false   true            [1]
