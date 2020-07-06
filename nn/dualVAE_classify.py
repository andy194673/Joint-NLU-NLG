import os
import sys
import random
import json
import time
import numpy as np
import torch
import torch.nn as nn
from encoder import SentEncoder
from decoder import Decoder
from nlu import NLU
from utils.criterion import NLLEntropy, NormKLLoss

class Hidden2Gaussian(nn.Module):
	def __init__(self, input_size, output_size, config):
		super(Hidden2Gaussian, self).__init__()
		self.mu = nn.Linear(input_size, output_size)
		self.logvar = nn.Linear(input_size, output_size)
		self.config = config

	def forward(self, enc_out, src, batch):
		'''
			enc_out: (B, T, H)
		'''
		if self.config.compute_z == 'mean':
			h = torch.mean(enc_out, dim=1)
		elif self.config.compute_z == 'last':
			h = enc_out[:, -1, :]
		mu, logvar = self.mu(h), self.logvar(h)
		return mu, logvar

	def sampleGaussian(self, mu, logvar, source):
		"""
		Sample from a multivariate Gaussian distribution
		Args:
			mu: (B, H)
			logvar: (B, H)
		"""
		if source == 'query': # no variance when input is query
			return mu

		epsilon = torch.randn(logvar.size()).cuda() # sample from N(0, 1) 
		std = torch.exp(0.5 * logvar)
		z = mu + std * epsilon
		return z
		

class DualVAE_classify(nn.Module):
	'''
	In NLU path, the encoder and decoder are an LSTM and several classifiers (defined in NLU module) respectively.
	In NLG path, both encoder and decoder are an LSTM.
	'''
	def __init__(self, config, dataset):
		super(DualVAE_classify, self).__init__()
		self.config = config
		self.dataset = dataset

		E, H = config.embed_size, config.hidden_size
		Z = config.latent_size
		D = config.dropout
		V = {}

		# model components
		self.encode = nn.ModuleDict({})
		if config.share_z:
			self.enc2lat = Hidden2Gaussian(2*H, Z, config)
			self.z_emb = nn.Linear(Z, H)
		else:
			self.enc2lat = nn.ModuleDict({})
			self.z_emb = nn.ModuleDict({})
		self.decode = nn.ModuleDict({})

		for src in ['query', 'parse']:
			V[src] = len(dataset.vocab[src])
			self.encode[src] = SentEncoder(V[src], E, H, dropout=D)
			if not config.share_z:
				self.enc2lat[src] = Hidden2Gaussian(2*H, Z, config)
				self.z_emb[src] = nn.Linear(Z, H)
			if src == 'query':
				self.decode[src] = Decoder(config, V[src], E, H, dataset.vocab[src], dataset.idx2word[src], \
									num_layers=config.num_layers, dropout=D, use_attn=config.attn, dec_len=config.dec_len)
			else:
				self.decode[src] = NLU(dataset, config, config.latent_size)
		
		# baseline estimator
#		self.bse = nn.Linear(H, 1)	

		self.gauss_kl = NormKLLoss(unit_average=False)
		self.set_optimizer()
		self._step = 0


#	def load_pretrain_lm(self):
#		self.pretrain_lm = {'query': None, 'parse': None}
#		if self.config.pretrain_query_lm != 'none':
#			assert os.path.exists(self.config.pretrain_query_lm)
#			lm = LM(self.config, self.dataset, 'query')
#			lm = lm.cuda()
#			lm.load_state_dict(torch.load(self.config.pretrain_query_lm))
#			self.pretrain_lm['query'] = lm

	def _forward(self, batch, mode='teacher_force', sample=False, beam_search=False, sources=list(['query', 'parse'])):
		'''
		Given (x, y) pair, reconstruct (x, y)
		NOTE: z is randomly sampled from either q(z|x) or q(z|y) in a batch
		'''
		# collect input tensor
		enc_input, enc_len, enc_mask = {}, {}, {}
		for src in ['query', 'parse']:
			enc_input[src] = batch['word_idx'][src] # (B, T), input is None if source is not provided
			enc_len[src] = batch['sent_len'][src] # (B,)
			enc_mask[src] = batch['mask'][src] # (B, T)
		batch_size = enc_input['query'].size(0) if enc_input['query'] is not None else enc_input['parse'].size(0)

		self.mu, self.logvar = {}, {}
		joint_output = {'query': {}, 'parse': {}}
		for src in sources:
			tgt = 'parse' if src == 'query' else 'query'

			# enc_out: (B, T, 2H) & state: tuple of (L, B, 2H)
			enc_output, enc_state = self.encode[src](enc_input[src], enc_len[src])

			if self.config.share_z:
				mu, logvar = self.enc2lat(enc_output, src, batch) # (B, Z)
				z = self.enc2lat.sampleGaussian(mu, logvar, src) # (B, Z)
				dec_init = self.z_emb(z).unsqueeze(0)
			else:
				mu, logvar = self.enc2lat[src](enc_output, src, batch) # (B, Z)
				z = self.enc2lat[src].sampleGaussian(mu, logvar, src) # (B, Z)
				dec_init = self.z_emb[src](z).unsqueeze(0)
			self.mu[src] = mu
			self.logvar[src] = logvar

			# test feeding zero here
#			print('feeding zero!', file=sys.stderr)
#			z = torch.zeros(*z.size()).cuda()
#			dec_init = torch.zeros(*dec_init.size()).cuda()
#			z = torch.randn(*z.size()).cuda()
#			dec_init = torch.randn(*dec_init.size()).cuda()
#			print('feeding random', file=sys.stderr)

			# decode
			if beam_search and enc_input[tgt] is None: # unsup 1st pass
				# autoencoder
				if src == 'query':
					output = self.decode[src](enc_input[src], (dec_init, dec_init), z, enc_output, enc_mask[src], \
								mode='teacher_force', sample=sample, source=src)
				else:
					output = self.decode[src](z)
				joint_output[src]['auto'] = output

				# obtain various outputs by either beam search or by sampling z TODO: sample z
				if tgt == 'query':
					output = self.decode[tgt].beam_search(enc_input[tgt], (dec_init, dec_init), z, enc_output, \
								enc_mask[src], beam_size=self.config.beam_size, source=src)
				else:
					output = self.decode[tgt](z, beam_search=True)
				joint_output[tgt]['trans'] = output

			elif beam_search and enc_input[tgt] is not None: # test w/i beam search
				if tgt == 'query':
					output = self.decode[tgt].beam_search(enc_input[tgt], (dec_init, dec_init), z, enc_output, \
								enc_mask[src], beam_size=self.config.beam_size, source=src)
				else:
					output = self.decode[tgt](z, beam_search=True)
				joint_output[tgt]['trans'] = output

			else: # sup training / test w/o beam search / unsup 2nd pass
				# autoencoder
				if batch['run_auto'][src]:
					if src == 'query':
						output = self.decode[src](enc_input[src], (dec_init, dec_init), z, enc_output, enc_mask[src], \
									mode=mode, sample=sample, source=src)
					else:
						output = self.decode[src](z)
					joint_output[src]['auto'] = output
				# translation
				if tgt == 'query':
					output = self.decode[tgt](enc_input[tgt], (dec_init, dec_init), z, enc_output, enc_mask[src], \
								mode=mode, sample=sample, source=src)
				else:
					output = self.decode[tgt](z) # NLU
				joint_output[tgt]['trans'] = output
		return joint_output


	def unsup_forward(self, batch, mode='teacher_force', beam_search=False):
		'''
		reconstruct the input source in unsupervised learning manner by running forward twice
		'''
		label_source, unlabel_source = self.checkLabelSource(batch)
#		print('label source: {} | unlabel source: {}'.format(label_source, unlabel_source))

		# first forward: generate y (or x) that is not given
		self.unsup_mu, self.unsup_logvar = {}, {}
		self.output = self._forward(batch, mode=mode, sample=False, beam_search=True, sources=list([label_source]))
		self.unsup_mu[label_source] = self.mu[label_source]
		self.unsup_logvar[label_source] = self.logvar[label_source]

		# prepare input tensor for second forward
		batch2 = self.sample2batch(self.output, batch, label_source, unlabel_source)
		self.batch2 = batch2

		# second forward: re-construct the input source x (or y) based on y_hat (or x_hat)
		t = time.time()
		self.output2 = self._forward(batch2, mode=mode, sample=False, beam_search=False, sources=list([unlabel_source]))
		self.unsup_mu[unlabel_source] = self.mu[unlabel_source]
		self.unsup_logvar[unlabel_source] = self.logvar[unlabel_source]


	def checkLabelSource(self, batch):
		'''
		check which source of information is given with label
		'''
		if batch['word_idx']['parse'] is not None and batch['word_idx']['query'] is not None:
			print('Both sources are not None')
			sys.exit(1)
		elif batch['word_idx']['parse'] is not None:
			label_source, unlabel_source = 'parse', 'query'
		elif batch['word_idx']['query'] is not None:
			label_source, unlabel_source = 'query', 'parse'
		else:
			print('Both sources are None')
			sys.exit(1)
		return label_source, unlabel_source


	def sample2batch(self, output, batch1, label_source, unlabel_source):
		'''
		Prepare input tensor for second forward
		Due to beam search, there are more than 1 hyp for an example. Manipulation of input tensor is needed.
		Two ways to form new input:
			1. form (B*beam_size, T) - cost more memory but forward only once (we use this!)
			2. keep (B, T) 			 - save memory but need to loop beam_size times on decoder
		'''
		# labelled source
		batch2 = {'word_idx': {}, 'sent_len': {}, 'mask': {}, 'ref': {}}
		batch_size, T = batch1['word_idx'][label_source].size()
		beam_size = self.config.beam_size
		for info in ['word_idx', 'mask']:
			batch2[info][label_source] = \
				batch1[info][label_source].unsqueeze(1).repeat(1, beam_size, 1).view(batch_size*beam_size, T)
		batch2['sent_len'][label_source] = batch1['sent_len'][label_source].unsqueeze(1).repeat(1, beam_size).view(batch_size*beam_size)
		batch2['ref'][label_source] =  [ref for ref in batch1['ref'][label_source] for _ in range(beam_size)]

		# unlabelled source
		# flatten (B, beam_size, T) -> (B'=B*beam_size, T)
		decode_len = [_len for batch_len in output[unlabel_source]['trans']['decode_len'] for _len in batch_len]
		max_len = max(decode_len)
		word_idx, mask = [], []
		for batch_idx in range(batch_size):
			for beam_idx in range(beam_size):
				sent = output[unlabel_source]['trans']['decode'][batch_idx][beam_idx].split()
				if len(sent) < self.config.dec_len:
					sent.append('<EOS>') # allign with supervised training
				s_len = len(sent)

				assert s_len == output[unlabel_source]['trans']['decode_len'][batch_idx][beam_idx] #or s_len == self.config.dec_len+1
				sent = [self.dataset.vocab[unlabel_source][word] for word in sent]
				sent.extend( [0 for _ in range(max_len-s_len)] ) # pad
				word_idx.append(sent)
				_mask = [0 for _ in range(s_len)] + [float('-inf') for _ in range(max_len-s_len)]
				mask.append(_mask)

		batch2['word_idx'][unlabel_source] = torch.tensor(word_idx).long().cuda()
		batch2['sent_len'][unlabel_source] = torch.tensor(decode_len).long().cuda()
		batch2['mask'][unlabel_source] = torch.tensor(mask).float().cuda()
		batch2['ref'][unlabel_source] = None
		batch2['run_auto'] = {label_source: False, unlabel_source: False}
		return batch2


	def forward(self, batch, mode='teacher_force', beam_search=False, sources=list(['query', 'parse'])):
		self.output = self._forward(batch, mode=mode, beam_search=beam_search, sources=sources)
		return self.output


	def get_loss(self, batch):
		'''
		loss for supervised learning
		'''
		auto, trans, kl = {}, {}, {}
		for tgt in ['query', 'parse']:
			src = 'parse' if tgt == 'query' else 'query'
			# autoencoder loss
			if tgt == 'query':
				auto[tgt] = NLLEntropy(self.output[tgt]['auto']['logits'], batch['word_idx'][tgt], ignore_idx=self.dataset.vocab[tgt]['<PAD>'])
			else:
				auto[tgt] = self.decode[tgt].get_loss(self.output[tgt]['auto']['output_dist'], batch['class'][tgt])

			# translation loss
			if tgt == 'query':
				trans[tgt] = NLLEntropy(self.output[tgt]['trans']['logits'], batch['word_idx'][tgt], ignore_idx=self.dataset.vocab[tgt]['<PAD>'])
			else:
				trans[tgt] = self.decode[tgt].get_loss(self.output[tgt]['trans']['output_dist'], batch['class'][tgt])

			# kl loss
			if self.config.compensate_prior:
				kl[tgt] = self.gauss_kl(self.mu[tgt], self.logvar[tgt], self.mu[src], self.logvar[src]) # (B, ) 
			else:
				kl[tgt] = self.gauss_kl(self.mu[tgt], self.logvar[tgt], torch.tensor(0).float(), torch.tensor(0).float()) # (B, ) 

		kl_w = self.get_kl_weight()
		auto_w = self.config.auto_weight
		loss = 0
		for tgt in ['query', 'parse']:
			loss += auto_w * auto[tgt] + trans[tgt] + kl_w * kl[tgt]

		self.output = self.mu = self.logvar = None # in case it's accidently used in unsupervised learning
		return {'auto1': auto['query'], 'auto2': auto['parse'], 'trans1': trans['query'], 'trans2': trans['parse'], \
				'kl': kl['query']+kl['parse'], 'kl_w': kl_w}, loss


	def get_unsup_loss(self, batch, n_unsup):
		def get_baseline(R):
			return torch.mean(R).detach() # should use detach(), just a negative number for each reward

		label_source, unlabel_source = self.checkLabelSource(batch)
		batch_size, T = batch['word_idx'][label_source].size()
		decode_len = self.output[unlabel_source]['trans']['decode_len']

		# pass the gradient through the generated data between two forwards using REINFORCE
		reward = self.get_reward_from_rec(label_source, batch) # (B, beam_size)
		baseline = get_baseline(reward)
		rewardDiff = reward
#		rewardDiff = reward - baseline # (B, T)
		
		# rl loss
		assert self.output[unlabel_source]['trans']['mode'] == 'gen'
		logprobs = self.output[unlabel_source]['trans']['logprobs'] # (B, T) or (B, beam_size) in beam_search
		rl_loss = torch.mean(-logprobs*rewardDiff)

		# rec loss
		assert self.output2[label_source]['trans']['mode'] == 'teacher_force'
		if label_source == 'query':
			index = batch['word_idx'][label_source].unsqueeze(1).repeat(1, self.config.beam_size, 1).view(batch_size*self.config.beam_size, T)
			rec_loss = NLLEntropy(self.output2[label_source]['trans']['logits'], index, ignore_idx=self.dataset.vocab[label_source]['<PAD>'])
		else:
			target = {}
			for slot in self.dataset.slot_list:
				target[slot] = \
					batch['class']['parse'][slot].unsqueeze(1).repeat(1, self.config.beam_size).view(batch_size*self.config.beam_size)
			rec_loss = self.decode['parse'].get_loss(self.output2['parse']['trans']['output_dist'], target)

		# auto-encoder loss
		if label_source == 'query':
			auto_loss = NLLEntropy(self.output[label_source]['auto']['logits'], batch['word_idx'][label_source], \
									ignore_idx=self.dataset.vocab[label_source]['<PAD>'])
		else:
			auto_loss = self.decode[label_source].get_loss(self.output[label_source]['auto']['output_dist'], batch['class'][label_source])

		# kl loss (B, Z) -> (beam_size*B, Z)
		self.unsup_mu[label_source] = self.unsup_mu[label_source].unsqueeze(1)\
						.repeat(1, self.config.beam_size, 1).view(batch_size*self.config.beam_size, self.config.latent_size)
		self.unsup_logvar[label_source] = self.unsup_logvar[label_source].unsqueeze(1)\
						.repeat(1, self.config.beam_size, 1).view(batch_size*self.config.beam_size, self.config.latent_size)
		kl_loss = 0
		kl_loss += self.gauss_kl(self.unsup_mu[label_source], self.unsup_logvar[label_source], \
									self.unsup_mu[unlabel_source], self.unsup_logvar[unlabel_source])
		kl_loss += self.gauss_kl(self.unsup_mu[unlabel_source], self.unsup_logvar[unlabel_source], \
									self.unsup_mu[label_source], self.unsup_logvar[label_source])
		# final loss
		auto_w = self.config.auto_weight
		kl_w = self.get_kl_weight()
		rec_w = self.config.rec_weight
		rl_w = self.config.rl_weight
		loss = auto_w*auto_loss + rl_w*rl_loss + rec_w*rec_loss + kl_w*kl_loss
		return {'auto': auto_loss, 'rec': rec_loss, 'rl': rl_loss, 'kl': kl_loss, 'exp_r': baseline}, loss


	def get_reward_from_rec(self, label_source, batch):
		batch_size, T = batch['word_idx'][label_source].size()
		beam_size = self.config.beam_size
		if label_source == 'query':
			# precision as reward
			reward = torch.zeros(batch_size, beam_size).float().cuda()
			count = torch.zeros(batch_size, beam_size).float().cuda()
			output_dist = self.output['parse']['trans']['output_dist'] # a dict of (B', C)
			for slot in self.dataset.slot_list:
				value, index = torch.max(torch.softmax(output_dist[slot].detach(), dim=1), dim=1) # (B')
				for batch_idx in range(batch_size):
					ref = batch['ref']['query'][batch_idx] # str
					for beam_idx in range(beam_size):
						value_idx = index[batch_idx*beam_size+beam_idx].item()
						if value_idx == len(self.dataset.ontology[slot]): # none
							continue
						count[batch_idx][beam_idx] += 1

						value = self.dataset.ontology[slot][value_idx]
						if value in ref:
							reward[batch_idx][beam_idx] += 1
			return reward/count # (B, beam_size)

		else: # label_source == 'parse'
			# nlu result as reward (binary or partial)
			output_dist = self.output2[label_source]['trans']['output_dist'] # a dict of (B', C)
			reward = torch.zeros(batch_size*beam_size).cuda() # (B')
			for slot in self.dataset.slot_list:
				target = batch['class']['parse'][slot].unsqueeze(1).repeat(1, beam_size).view(-1) # (B')
				value, index = torch.max(torch.softmax(output_dist[slot].detach(), dim=1), dim=1) # (B')
				reward += (index == target).float()
			reward /= len(self.dataset.slot_list)
			reward = reward.view(batch_size, beam_size)
			return reward


	def get_kl_weight(self):
		train_len = len(self.dataset.data['train'])
		n_epoch = self.config.kl_epoch
		if train_len % self.config.batch_size == 0:
			updates_in_epoch = train_len//self.config.batch_size
		else:
			updates_in_epoch = train_len//self.config.batch_size + 1
		period_len = n_epoch * updates_in_epoch
		if self.config.kl_anneal_type == 'monotonic':
			return min(1, float(self._step/period_len))

		elif self.config.kl_anneal_type == 'cycle':
			R = 0.5
			return min(1, float((self._step % period_len)/(period_len*R)))

		else: # no annealing at all
			return 0.5

	def update(self, loss, msg):
		assert msg == 'supervised' or msg == 'unsupervised'
		loss.backward()
		grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
		if msg == 'supervised':
			self.optimizer.step()
			self.optimizer.zero_grad()
			self._step += 1
		else:
			self.unsup_optimizer.step()
			self.unsup_optimizer.zero_grad()
		return grad_norm

	def set_optimizer(self):
		if self.config.optimizer == 'adam':
			if self.config.mode == 'pretrain':
				self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.lr)
			elif self.config.mode == 'finetune':
				self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.unsup_lr)
			self.unsup_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), \
					lr=self.config.unsup_lr)
		else:
			raise NotImplementedError

	def saveModel(self, epoch):
		if not os.path.exists(self.config.model_dir):
			os.makedirs(self.config.model_dir)
		torch.save(self.state_dict(), self.config.model_dir + '/epoch-{}.pt'.format(str(epoch)))
		torch.save(self.state_dict(), self.config.model_dir + '/epoch-{}.pt'.format('best'))

	def loadModel(self, epoch):
		model_name = self.config.model_dir + '/epoch-{}.pt'.format(str(epoch))
		self.load_state_dict(torch.load(model_name))

	def sample_mu(self, batch, mode='teacher_force', beam_search=False, sources=list(['query', 'parse'])):
		_ = self._forward(batch, mode=mode, beam_search=beam_search, sources=sources)
		return self.mu['query'], self.mu['parse']
