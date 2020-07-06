import sys
import argparse
import time
import os
import json
import random
import torch
import numpy as np
from data import DataLoader
from utils.metric import weather_metric, e2e_metric
sys.path.insert(0, './nn')
from dualVAE import DualVAE
from dualVAE_classify import DualVAE_classify

def	update_once_unsup(dataset, model, log, unsup_type):
	'''
	update the model once using unsupervised learning
	'''
	t0 = time.time()
	accumulate_loss = 0
	for _step in range(config.unsup_step_for_update):
		unsup_batch = dataset.next_batch(unsup_type)
		log[unsup_type]['n'] += 1

		model.unsup_forward(unsup_batch, mode='teacher_force', beam_search=True)
		loss_log, batch_loss = model.get_unsup_loss(unsup_batch, log[unsup_type]['n'])

		for k, v in loss_log.items(): log[unsup_type][k] += loss_log[k]
		accumulate_loss += batch_loss
	accumulate_loss /= config.unsup_step_for_update
	grad_norm = model.update(accumulate_loss, 'unsupervised')
	log[unsup_type]['grad'] += grad_norm

	if log['epoch_idx'] == 1 and log[unsup_type]['n'] <= config.unsup_step_for_update:
		print('{} | grad norm: {:.3f} | time for updating once: {:.2f}'.format(unsup_type, grad_norm, time.time()-t0),file=sys.stderr)

def	update_once_sup(dataset, model, log):
	'''
	update the model once using supervised learning
	'''
	t0 = time.time()

	# draw a batch
	batch = dataset.next_batch('train')
	if batch == None:
		return False
	if config.mode == 'finetune' and not config.unsup_alter_sup: # don't really update the model
		return True
	log['sup']['n'] += 1

	# forward
	_ = model(batch, mode='teacher_force')
	loss_log, loss = model.get_loss(batch)
	
	# update
	grad_norm = model.update(loss, 'supervised')

	# collect log
	log['sup']['grad'] += grad_norm
	for k, v in loss_log.items(): log['sup'][k] += loss_log[k]

	if log['sup']['n'] == 1 and log['epoch_idx'] == 1:
		print('1 batch takes {:.1f} sec, estimated time for a epoch: {:.1f}'.format(time.time()-t0, \
				len(dataset.data['train'])/config.batch_size*(time.time()-t0) ),file=sys.stderr)
	return True # to signify the update is successful


def test_model(dType, epoch_idx, model, mode, beam_search=True):
	t0 = time.time()
	n, acc1, acc2 = 0, 0, 0
	hyps = {'parse': [], 'query': []}
	refs = {'parse': [], 'query': []}
	while True:
		# get a testing batch
		batch = dataset.next_batch(dType)
		n += 1
		if batch == None:
			break
		output = model(batch, mode='gen', beam_search=beam_search)
				
		# collect output for evaluation
		for source in config.dec_side:
			for ref, hyp in zip(batch['ref'][source], output[source]['trans']['decode']):
				refs[source].append(ref)
				if not beam_search:
					hyps[source].append(hyp)
				else:
					hyps[source].append(hyp[0]) # take top1 beam in beam search

	# print testing log
	t_spent = time.time()-t0
	if dType == 'test': dType = 'test '
	if config.dataset == 'weather':
		parse_acc, query_bleu, query_bleu_group, output = weather_metric(hyps, refs, config, beam_search, dType)
		print('{} Epoch {} | Parse Acc: {:.2f}% | Query bleu {:.3f} | time: {:.1f} (beam_search={})'.format(dType, epoch_idx, \
				parse_acc, query_bleu, query_bleu_group, t_spent, beam_search))
		print('{} Epoch {} | Parse Acc: {:.2f}% | Query bleu {:.3f} | time: {:.1f} (beam_search={})'.format(dType, epoch_idx, \
				parse_acc, query_bleu, t_spent, beam_search),file=sys.stderr)
		return parse_acc, parse_acc, query_bleu, query_bleu, output
	elif config.dataset == 'e2e':
		slot_f1, value_f1, joint_acc, query_bleu, str_match_acc, output = e2e_metric(hyps, refs, config, beam_search, dType, dataset)
		print('{} Epoch {} | joint acc: {:.2f}% | Value f1: {:.3f} | Query bleu {:.3f} | Semantic Acc: {:.2f}% | time: {:.1f} (beam_search={})' \
				.format(dType, epoch_idx, joint_acc, value_f1, query_bleu, str_match_acc, t_spent, beam_search))
		print('{} Epoch {} | joint acc: {:.2f}% | Value f1: {:.3f} | Query bleu {:.3f} | Semantic Acc: {:.2f}% | time: {:.1f} (beam_search={})' \
				.format(dType, epoch_idx, joint_acc, value_f1, query_bleu, str_match_acc, t_spent, beam_search),file=sys.stderr)
		return value_f1, joint_acc, query_bleu, str_match_acc, output


def trainOneEpoch(dType, epoch_idx, model):
	'''
	train the model for one epoch
	'''
	t0 = time.time()
	log = {'sup': {'grad': 0, 'n': 0, 'auto1': 0, 'auto2': 0, 'trans1': 0, 'trans2': 0, 'kl': 0, 'kl_w': 0}, \
  			'unsup_query': {'grad': 0, 'n': 0, 'auto': 0, 'rec': 0, 'rl': 0, 'kl': 0, 'exp_r': 0}, \
			'unsup_parse': {'grad': 0, 'n': 0, 'auto': 0, 'rec': 0, 'rl': 0, 'kl': 0, 'exp_r': 0}, \
			'epoch_idx': epoch_idx}

	# source of unsupervised learning
	if config.unsup_source == 'both':
		unsup_source = ['unsup_parse', 'unsup_query']
	else:
		unsup_source = ['unsup_{}'.format(config.unsup_source)]

	while True:
		# iterate between supervised learning and unsupervised learning if specified
		update_success = update_once_sup(dataset, model, log)
		if not update_success:
			break

		if config.unsup_learn and epoch_idx >= config.unsup_epoch:
			for _ in range(config.unsup_update_ratio):
				for unsup_type in unsup_source:
					update_once_unsup(dataset, model, log, unsup_type=unsup_type)

	# print the training log
	t_spent = time.time()-t0
	log1 = log['sup']
	n = log['sup']['n']
	if config.mode == 'pretrain' or config.unsup_alter_sup:
		print('{} Epoch {} | Loss Auto1 {:.3f}, Auto2 {:.3f}, Trans1 {:.3f}, Trans2 {:.3f}, kl {:.3f}, kl_w {:.2f} | GRAD: {:.2f} | time: {:.1f}'\
			.format('train', epoch_idx, log1['auto1']/n, log1['auto2']/n, log1['trans1']/n, log1['trans2']/n, log1['kl']/n, log1['kl_w']/n, log1['grad']/n, t_spent))
		print('{} Epoch {} | Loss Auto1 {:.3f}, Auto2 {:.3f}, Trans1 {:.3f}, Trans2 {:.3f}, kl {:.3f}, kl_w {:.2f} | GRAD: {:.2f} | time: {:.1f}'\
			.format('train', epoch_idx, log1['auto1']/n, log1['auto2']/n, log1['trans1']/n, log1['trans2']/n, log1['kl']/n, log1['kl_w']/n, log1['grad']/n, t_spent),file=sys.stderr)

	if config.unsup_learn and epoch_idx >= config.unsup_epoch:
		for unsup in unsup_source:
			log2 = log[unsup]
			n = log[unsup]['n']
			print('{} | Loss Auto: {:.3f} | Rec {:.3f}, RL {:.3f}, kl {:.3f} exp_R {:.3f} | GRAD: {:.2f}'.format(unsup, \
				log2['auto']/n, log2['rec']/n, log2['rl']/n, log2['kl']/n, log2['exp_r']/n, log2['grad']/n*config.unsup_step_for_update))
			print('{} | Loss Auto: {:.3f} | Rec {:.3f}, RL {:.3f}, kl {:.3f} exp_R {:.3f} | GRAD: {:.2f}'.format(unsup, \
				log2['auto']/n, log2['rec']/n, log2['rl']/n, log2['kl']/n, log2['exp_r']/n, log2['grad']/n*config.unsup_step_for_update), file=sys.stderr)

def trainIter(config, dataset, model):
	'''
	training for specific epochs
	'''
	# load model as init for finetuning
	if config.mode == 'finetune':
		print('The performance before finetune')
		test(config, dataset, model)
		model.loadModel(config.load_epoch)

	result = {}
	no_improve_count = 0
	best_score = 0
	for epoch_idx in range(1, config.epoch+1):
		dataset.init()
		# train
		model.train()
		trainOneEpoch('train', epoch_idx, model)

		# evaluate
		model.eval()
		rec = {}
		result[epoch_idx] = {}
		with torch.no_grad():
			for dType in ['valid', 'test']:
				dataset.init()
				rec[dType] = {}
				rec[dType]['NLU-f1'], rec[dType]['NLU-acc'], rec[dType]['NLG-bleu'], rec[dType]['NLG-acc'], _ = \
					test_model(dType, epoch_idx, model, mode='gen', beam_search=False)
				result[epoch_idx][dType] = rec[dType]

				# pick the best model by the average of NLU and NLG results
				if dType == 'valid':
					score = 0.5*(rec[dType]['NLU-f1'] + rec[dType]['NLG-bleu'])

		with open(config.result_path, 'w') as f:
			json.dump(result, f, indent=4, sort_keys=True)

		# select model by valid result with beam search & save model
		if score > best_score:
			best_score = score
			no_improve_count = 0
			print('Best score on validation!')
			print('Best score on validation!',file=sys.stderr)
			model.saveModel(str(epoch_idx))
		else:
			no_improve_count += 1			

		# early stopping
		if no_improve_count >= config.no_improve_epoch:
			print('Early stop!')
			print('Early stop!',file=sys.stderr)
			sys.exit(1)
		print('---------------------------------------------------------')
		print('---------------------------------------------------------',file=sys.stderr)


def test(config, dataset, model):
	# test
	model.loadModel(config.load_epoch)
	epoch_idx = config.load_epoch
	model.eval()
	rec = {}
	decode = {'result': {}}
	with torch.no_grad():
		for dType in ['valid', 'test']:
			dataset.init()
			rec = {}
			rec['NLU-f1'], rec['NLU-acc'], rec['NLG-bleu'], rec['NLG-acc'], DECODE = \
				test_model(dType, epoch_idx, model, mode='gen', beam_search=False)
			decode['result']['{}'.format(dType)] = rec
			decode['{}'.format(dType)] = DECODE

			# turn on beam search
			dataset.init()
			rec = {}
			rec['NLU-f1'], rec['NLU-acc'], rec['NLG-bleu'], rec['NLG-acc'], DECODE = \
				test_model(dType, epoch_idx, model, mode='gen', beam_search=True)
			decode['result']['{} (beam)'.format(dType)] = rec
			decode['{} (beam)'.format(dType)] = DECODE

	# write decode result
	with open(config.decode_path, 'w') as f:
		json.dump(decode, f, indent=4, sort_keys=True)


def get_config(): # TODO: clean config when checking hyper-parameters
	def str2bool(v):
		if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
			return True
		elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
			return False
		else:
			raise argparse.ArgumentTypeError('Boolean value expected.')

	parser = argparse.ArgumentParser(description='')
	# data/model path
	parser.add_argument('--dataset', type=str, required=True)
	parser.add_argument('--train_path', type=str, default='none')
	parser.add_argument('--valid_path', type=str, required=True)
	parser.add_argument('--test_path', type=str, required=True)
	parser.add_argument('--word2count_query', type=str, required=True, help='count of token in natural language, which will be used in building vocab')
	parser.add_argument('--word2count_parse', type=str, required=True, help='count of token in meaning representation, which will be used in building vocab')
	parser.add_argument('--model_dir', type=str, required=True)
	parser.add_argument('--result_path', type=str, default='none')
	parser.add_argument('--decode_path', type=str, default='none')

	# general setup
	parser.add_argument('--mode', type=str, required=True)
	parser.add_argument('--vocab_size', type=int, required=True)
	parser.add_argument('--shuffle', type=str2bool, default=True, help='whether to shuffle training dataset during training')
	parser.add_argument('--batch_size', type=int, required=True, default=32, help='number of examples in a batch')
	parser.add_argument('--beam_size', type=int, default=3, help='beam search size')
	parser.add_argument('--dec_len', type=int, default=100, help='max decoding length')
	parser.add_argument('--grad_clip', type=float, default=5)
	parser.add_argument('--optimizer', type=str, default='adam')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate of supervised learning')
	parser.add_argument('--epoch', type=int, default=100, help='training epochs')
	parser.add_argument('--seed', type=int, default=1122)
	parser.add_argument('--kl_anneal_type', type=str, default='monotonic', help='kl annealing type')
	parser.add_argument('--kl_epoch', type=int, default=100)
	parser.add_argument('--enc_side', type=list, default=['parse', 'query'])
	parser.add_argument('--dec_side', type=list, default=['parse', 'query'])
	parser.add_argument('--no_improve_epoch', type=int, default=10)
	parser.add_argument('--load_epoch', type=str, default='best')

	# dataset dependent parameters
	parser.add_argument('--ontology', type=str, default='data/e2e/ontology.json', help='ontology of e2e dataset')

	# unsupervised learning setup		
	parser.add_argument('--unsup_learn', type=str2bool, default=False, help='whether to use unsupervised learn for training')
	parser.add_argument('--unsup_epoch', type=int, default=1, help='the epoch where unsupervised learning starts at')
	parser.add_argument('--unsup_query_path', type=str, default='none')
	parser.add_argument('--unsup_parse_path', type=str, default='none')
	parser.add_argument('--unsup_update_ratio', type=int, default=1, help='unsupervised updates ratio against 1 supervised update')
	parser.add_argument('--unsup_reward_type', type=str, default='likelihood', help='reward type used in REINFORCE for unsupervised learning')
	parser.add_argument('--unsup_step_for_update', type=int, default=1, help='accumulate gradient for how many step (batch) for 1 update')
	parser.add_argument('--unsup_source', type=str, default='both', help='unsupervised sources')
	parser.add_argument('--unsup_lr', type=float, default=0.0001)
	parser.add_argument('--unsup_alter_sup', type=str2bool, default=True, help='whether to alterate between supervised and unsupervised')

	# model hyper-parameter
	parser.add_argument('--embed_size', type=int, default=300)
	parser.add_argument('--hidden_size', type=int, default=300)
	parser.add_argument('--latent_size', type=int, default=300)
	parser.add_argument('--dropout', type=float, default=0.2)
	parser.add_argument('--num_layers', type=int, default=1)
	parser.add_argument('--bidirectional', type=str2bool, default=True)
	parser.add_argument('--attn', type=str2bool, default=True)
	parser.add_argument('--peep', type=str2bool, default=True)
	parser.add_argument('--share_z', type=str2bool, default=True)
	parser.add_argument('--compensate_prior', type=str2bool, default=True)
	parser.add_argument('--auto_weight', type=float, default=0.5)
	parser.add_argument('--rec_weight', type=float, default=0.5)
	parser.add_argument('--rl_weight', type=float, default=1)

	# hyper-parameter related to latent variable
	parser.add_argument('--dropout_attn_level', type=str, default='unit')
	parser.add_argument('--dropout_attn_prob', type=float, required=True)
	parser.add_argument('--compute_z', type=str, required=True)
	parser.add_argument('--nlu_embed', type=str2bool, default=False)
	parser.add_argument('--none_weight', type=float, default=1, help='loss of weight of not_mention class in e2e nlu')

	args = parser.parse_args()
	if args.unsup_learn:
		assert args.unsup_epoch >= 0
		assert os.path.exists(args.unsup_query_path) or os.path.exists(args.unsup_parse_path)
		assert args.unsup_update_ratio >= 1
		assert args.unsup_source in ['both', 'parse', 'query']
	assert args.mode in ['pretrain', 'test', 'finetune']
	assert args.dataset in ['e2e', 'weather']
	assert args.dropout_attn_level in ['step', 'unit']
	assert args.compute_z in ['mean', 'last']

	if args.mode == 'pretrain':
		assert args.train_path != 'none'
		assert args.result_path != 'none'
		assert args.kl_anneal_type in ['monotonic', 'cycle', 'none']
	if args.mode == 'finetune':
		assert args.train_path != 'none'
		assert args.result_path != 'none'
	if args.mode == 'test':
		assert args.decode_path != 'none'
	print(args)
	return args

def set_seed(args):
	'''
	for reproduction
	'''
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
	# load config
	config = get_config()
	set_seed(config)

	# load data
	dataset = DataLoader(config)

	# construct models, different model structure for different dataset
	if config.dataset == 'e2e':
		model = DualVAE_classify(config, dataset)
	else:
		model = DualVAE(config, dataset)
	model = model.cuda()
	
	# start training / testing
	if config.mode == 'pretrain' or config.mode == 'finetune':
		trainIter(config, dataset, model)
	else: # test
		test(config, dataset, model)
