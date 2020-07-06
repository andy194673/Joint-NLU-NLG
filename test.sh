#!/bin/bash
dataset=$1 # e2e / weather
ratio=$2 # data ratio, 5 10 25 50 100
batch_size=$3 # 32 / 64
dim=$4 # 150 / 300
model_name=$5

# data path
train_path='data/'$dataset'_ratio/'$ratio'/train.tsv'
valid_path='data/'$dataset'/dev.tsv'
test_path='data/'$dataset'/test.tsv'
word2count_mr='data/'$dataset'/word2count_mr.json' # word count of meaning representation
word2count_nl='data/'$dataset'/word2count_nl.json' # word count of natural language

# some dataset-dependent hyper-parameters as default
if [ "$dataset" = "e2e" ]; then
	vocab=1200
	auto_weight=0
	dropout_attn_prob=0.9
	compute_z='mean'
	peep='True'
elif [ "$dataset" = "weather" ]; then
	vocab=500
	auto_weight=1
	dropout_attn_prob=0
	compute_z='last'
	peep='False'
fi

####### test the model trained using only supervised learning #######
model_dir='checkpoint/pretrain/'$model_name
decode='res/decode/pretrain/'$model_name'.json' # decode output
python3 main.py --dataset=$dataset --mode='test' --model_dir=$model_dir --batch_size=$batch_size \
				--train_path=$train_path --valid_path=$valid_path --test_path=$test_path \
				--word2count_query=$word2count_nl --word2count_parse=$word2count_mr --vocab_size=$vocab \
				--embed_size=$dim --hidden_size=$dim --latent_size=$dim \
				--compute_z=$compute_z --peep=$peep --dropout_attn_prob=$dropout_attn_prob \
				--decode_path=$decode > /dev/null


####### test the model trained using semi-supervised learning #######
model_dir='checkpoint/finetune/'$model_name
decode='res/decode/finetune/'$model_name'.json' # decode output
python3 main.py --dataset=$dataset --mode='test' --model_dir=$model_dir --batch_size=$batch_size \
				--train_path=$train_path --valid_path=$valid_path --test_path=$test_path \
				--word2count_query=$word2count_nl --word2count_parse=$word2count_mr --vocab_size=$vocab \
				--embed_size=$dim --hidden_size=$dim --latent_size=$dim \
				--compute_z=$compute_z --peep=$peep --dropout_attn_prob=$dropout_attn_prob \
				--decode_path=$decode > /dev/null
