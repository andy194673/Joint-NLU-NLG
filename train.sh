#!/bin/bash
dataset=$1 # e2e / weather
ratio=$2 # data ratio, 5 10 25 50 100
batch_size=$3 # 32 / 64
dim=$4 # 150 / 300
seed=$5

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

####### pretrain the model using supervised learning first #######
mode='pretrain'
model_name=$dataset'_ratio'$ratio'_seed'$seed
model_dir='checkpoint/'$mode'/'$model_name # path of the stored model
result='res/result/'$mode'/'$model_name'.json' # result of each epoch
log='res/log/'$mode'/'$model_name'.log' # training log
python3 main.py --dataset=$dataset --mode=$mode --batch_size=$batch_size --seed=$seed \
				--train_path=$train_path --valid_path=$valid_path --test_path=$test_path \
				--word2count_query=$word2count_nl --word2count_parse=$word2count_mr --vocab_size=$vocab \
				--embed_size=$dim --hidden_size=$dim --latent_size=$dim \
				--epoch=100 --no_improve_epoch=10 --model_dir=$model_dir --result_path=$result \
				--auto_weight=$auto_weight --compute_z=$compute_z --peep=$peep --dropout_attn_prob=$dropout_attn_prob > $log


####### finetune the model using semi-supervised learning #######
mode='finetune'
fine_model_dir='checkpoint/'$mode'/'$model_name
result='res/result/'$mode'/'$model_name'.json'
log='res/log/'$mode'/'$model_name'.log'
mkdir -p $fine_model_dir
cp $model_dir'/epoch-best.pt' $fine_model_dir'/epoch-best.pt'
unlabel_nl='data/'$dataset'_ratio/'$ratio'/unlabel_nl.tsv' # can also use reuse_nl.tsv for reusing labelled data
unlabel_mr='data/'$dataset'_ratio/'$ratio'/unlabel_mr.tsv' # can also use reuse_mr.tsv for reusing labelled data
unsup_source='both'
python3 main.py --dataset=$dataset --mode=$mode --batch_size=$batch_size --seed=$seed \
				--train_path=$train_path --valid_path=$valid_path --test_path=$test_path \
                --word2count_query=$word2count_nl --word2count_parse=$word2count_mr --vocab_size=$vocab \
                --embed_size=$dim --hidden_size=$dim --latent_size=$dim --kl_anneal_type='none' \
                --epoch=150 --no_improve_epoch=50 --model_dir=$fine_model_dir --result_path=$result \
                --auto_weight=$auto_weight --rl_weight=1 --rec_weight=1 \
				--compute_z=$compute_z --peep=$peep --dropout_attn_prob=$dropout_attn_prob \
                --unsup_learn=true --unsup_source=$unsup_source --unsup_query_path=$unlabel_nl --unsup_parse_path=$unlabel_mr > $log
