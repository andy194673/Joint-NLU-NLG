## A Generative Model for Joint Natural Language Understanding and Generation
The source code of the paper [A Generative Model for Joint Natural Language Understanding and Generation](https://arxiv.org/abs/2006.07499) published at ACL 2020.

	@inproceedings{tseng2020generative,
	  title={A Generative Model for Joint Natural Language Understanding and Generation},
	  author={Tseng, Bo-Hsiang and Cheng, Jianpeng and Fang, Yimai and Vandyke, David},
	  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
	  pages={1795--1807},
	  year={2020}
	}

### Requirements
	python 3
	torch 1.1.0
	numpy 1.13.3
	nltk 3.4.5

### Data
The two used dataset are in data/ folder with different amounts of training data.

### Training
To train the model, use the script train.sh with the following command:

	bash train.sh $dataset $data_ratio $batch_size $model_dimension $seed

- dataset: e2e or weather
- data ratio: 5 / 10 / 25 / 50 / 100
- batch size: 32 / 64
- model dimension: 150 / 300

### Testing
	To test the model, use the script test.sh with the following command:
	bash test.sh $dataset $data_ratio $batch_size $model_dimension $model_name
	- dataset: e2e or weather
	- data ratio: 5 / 10 / 25 / 50 / 100
	- batch size: 32 / 64
	- model dimension: 150 / 300
	- model name: name of a trained model
