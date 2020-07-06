import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from attention import Attn

class Decoder(nn.Module):
	def __init__(self, config, input_size, embed_size, hidden_size, vocab, idx2word, num_layers=1, dropout=0.5, use_attn=True, use_peep=True, dec_len=50):
		super(Decoder, self).__init__()
		self.config = config
		self.input_size = input_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.output_size = input_size

		self.dropout = nn.Dropout(p=dropout)
		self.embed = nn.Embedding(input_size, embed_size)
		self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, \
						dropout=dropout, bidirectional=False, batch_first=True)
		self.use_attn = use_attn
		self.use_peep = use_peep

		if use_attn:
			self.dropout_on_attn = nn.Dropout(p=self.config.dropout_attn_prob)
			self.attn = nn.ModuleDict({
				'query': Attn(2*hidden_size, hidden_size),
				'parse': Attn(2*hidden_size, hidden_size)
			})
			if use_peep: # whether to use latent variable z at each time step
				self.out = nn.ModuleDict({
					'query': nn.Linear(3*hidden_size+self.config.latent_size, self.output_size),
					'parse': nn.Linear(3*hidden_size+self.config.latent_size, self.output_size)
				})
			else:
				self.out = nn.ModuleDict({
					'query': nn.Linear(3*hidden_size, self.output_size),
					'parse': nn.Linear(3*hidden_size, self.output_size)
				})
		else:
			if use_peep:
				self.out = nn.Linear(hidden_size+self.config.latent_size, self.output_size)
			else:
				self.out = nn.Linear(hidden_size, self.output_size)

		self.word2idx = vocab
		self.idx2word = idx2word
		self.dec_len = dec_len


	def _step(self, input_emb, init_state, latent_var, enc_output, enc_mask, hiddens, t, source):
		output, state = self.rnn(input_emb, init_state) # (B, 1, H) & (L, B, H)
		if self.use_attn:
			attn_dist, ctx_vec = self.attn[source](output, enc_output, enc_mask) # (B, T) & (B, H)
			if self.mode == 'teacher_force': # only dropout attention during training
				if self.config.dropout_attn_level == 'step' and \
					(random.randint(0, 100000) % 100+1) / 100 <= self.config.dropout_attn_prob:
					ctx_vec = torch.zeros(*ctx_vec.size()).cuda()
				elif self.config.dropout_attn_level == 'unit':
					ctx_vec = self.dropout_on_attn(ctx_vec)
			output = torch.cat([output.squeeze(1), ctx_vec], dim=1) # (B, V)
		else:
			output = output.squeeze(1)
		if self.use_peep:
			output = torch.cat([output, latent_var], dim=1)

		if self.use_attn:
			output = self.out[source](output) # (B, V)
		else:
			output = self.out(output) # (B, V)
		return output, state

	def forward(self, input_var, init_state, latent_var, enc_output, enc_mask, mode='teacher_force', sample=False, source='none'):
		'''
		Greedy search
		Args:
			input_var: (B, T)
			init_state: tuple of (L, B, H)
			latent_var: (B, H) can be the sample of z or just the mean 
		Return:
			output_prob: (B, T, V)
		'''
		self.batch_size = init_state[0].size(1)
		max_len = self.dec_len if mode == 'gen' else input_var.size(1)
		assert mode == 'teacher_force' or mode == 'gen'
		assert source in ['query', 'parse']
		self.mode = mode

		if not isinstance(init_state, tuple): # only for lstm
			init_state = tuple([init_state.unsqueeze(0), init_state.clone().unsqueeze(0)])
			print('should not enter here')
			sys.exit(1)

		go_idx = torch.tensor([self.word2idx['<SOS>'] for b in range(self.batch_size)]).long().cuda() # (B, )
		input_emb = self.dropout(self.embed(go_idx)).unsqueeze(1) # (B, 1, E)

		# output container
		logits = torch.zeros(self.batch_size, max_len, self.output_size).cuda()
		logprobs = torch.zeros(self.batch_size, max_len).cuda()
		hiddens = torch.zeros(self.batch_size, max_len, self.hidden_size).cuda() if mode == 'gen' else None # (B, T, H)
		sample_wordIdx = torch.zeros(self.batch_size, max_len).long().cuda() if mode == 'gen' else None
		sentences = [[] for b in range(self.batch_size)] if mode == 'gen' else None
		finish_flag = [0 for _ in range(self.batch_size)] if mode == 'gen' else None

		# generate output sequence step by step
		for t in range(max_len):
			output, state = self._step(input_emb, init_state, latent_var, enc_output, enc_mask, None, t, source) # (B, V)
			logits[:, t, :] = output
			if mode == 'gen': # only sample when gen for speedup training
				self.logits2words(output, sentences, sample_wordIdx, logprobs, t, finish_flag, sample) # collect ouput word at each time step
			if mode == 'gen' and sum(finish_flag) == self.batch_size: # break if all sentences finish
				break

			if mode == 'teacher_force':
				idx = input_var[:, t]
			else:
				value, idx = torch.max(output, dim=1) # (B, )
			input_emb = self.dropout(self.embed(idx)).unsqueeze(1) # (B, 1, E)
			init_state = state

		if mode == 'gen':
			sentences_len = [len(sent) for sent in sentences] # sentence length w/i eos
			sentences = [' '.join(sent[:-1]) for sent in sentences] # remove eos and convert to string
			# pad 0 in sample_wordIdx for generating samples
			for b, sent_len in enumerate(sentences_len):
				if sent_len < max_len:
					assert sample_wordIdx[b, sent_len-1] == self.word2idx['<EOS>']
				sample_wordIdx[b, sent_len:] = 0
				logprobs[b, sent_len:] = 0

#		return logits, sentences
		if mode == 'gen':
			output = {'logits': logits, 'logprobs': logprobs, 'sample_wordIdx': sample_wordIdx, \
						'decode': sentences, 'decode_len': sentences_len, 'hiddens': hiddens, 'mode': mode}
		else:
			output = {'logits': logits, 'logprobs': logprobs, 'sample_wordIdx': None, \
						'decode': None, 'decode_len': None, 'hiddens': None, 'mode': mode}
		return output
	

	def logits2words(self, logits, sentences, sample_wordIdx, logprobs, t, finish_flag, sample):
		'''
		logits: (B, V)
		sample_wordIdx, logprobs: (B, T)
		'''
		if sample:
			T = 2 # temperature, > 0 to encourage explore
			probs = torch.softmax(logits/T, dim=1)
			cat = Categorical(probs)
			idx = cat.sample() # (B, )
			value = torch.gather(torch.softmax(logits, dim=1), dim=1, index=idx.unsqueeze(1)) # (B, )
			value = value.squeeze(1)
		else:
			value, idx = torch.max(torch.softmax(logits, dim=1), dim=1) # (B, )
		
		sample_wordIdx[:, t] = idx
		logprobs[:, t] = torch.log(value)
		for b_idx, (sentence, i) in enumerate(zip(sentences, idx)):
			if len(sentence) > 0 and sentence[-1] == '<EOS>':
				finish_flag[b_idx] = 1
				continue
			sentence.append(self.idx2word[i.item()])


	def beam_search(self, input_var, init_state, latent_var, enc_output, enc_mask, beam_size=10, source=None):
		'''
		The optimized beam search in terms of speed for pytorch
		'''
		t0 = time.time()
		self.mode = 'gen'
		assert source in ['query', 'parse']
		t_input, t_cand, t_ff, t_sort, t_copy = 0, 0, 0, 0, 0
		t_cons, t_copy, t_his, t_log, t_state, t_pool, t_idx, t_beam = 0,0,0,0,0,0,0,0
		t_beam_size, t_left_size = 0, 0
		t_input1, t_input2 = 0, 0
		class Beam(object):
			def __init__(self, src_beam, state):
				if src_beam is not None:
					self.history = list(src_beam.history)
					self.logprob = src_beam.logprob.clone()
				else:
					self.history = ['<SOS>']
					self.logprob = torch.tensor(0).float()
					self.state = state # (L, 1, H)

		assert isinstance(init_state, tuple)
		batch_size = init_state[0].size(1)
		alpha = 0.7 # length normalization coefficient
		global_pool = [ [ Beam(None, (init_state[0][:, b, :].unsqueeze(1), init_state[1][:, b, :].unsqueeze(1))) ] \
							for b in range(batch_size)]

		for step_t in range(self.dec_len):
			cand_pool = [ [] for _ in range(batch_size) ]
			logprob_pool =  [ [] for _ in range(batch_size) ] # list for only logprob of beams in cand_pool

			# put finished beam (ending with eos) directly to candidate pool and remove them from the input to next ff
			if step_t > 0:
				trim_pool = []
				for batch_idx in range(batch_size):
					_pool = []
					for beam_idx in range(beam_size):
						beam = global_pool[batch_idx][beam_idx]
						if beam.history[-1] == '<EOS>':
							cand_pool[batch_idx].append(beam)
							logprob_pool[batch_idx].append(beam.logprob)
						else:
							_pool.append(beam)
					trim_pool.append(_pool)
			else:
				trim_pool = global_pool

			num_leftBeam = [len(beams) for beams in trim_pool] # number of left beams in each example
			new_batch_size = sum(num_leftBeam)
			if new_batch_size == 0:
				break

			# run a rnn step
			trim_pool = [beam for beams in trim_pool for beam in beams] # flatten trim pool for creating a batch
			input_idx = torch.tensor([self.word2idx[ trim_pool[l].history[-1] ] \
							for l in range(new_batch_size)]).long().cuda() # (B, )
			input_emb = self.dropout(self.embed(input_idx)).unsqueeze(1) # (B, 1, E)

			init_h = torch.cat([ trim_pool[l].state[0] for l in range(new_batch_size) ], dim=1) # (L, B, H)
			init_c = torch.cat([ trim_pool[l].state[1] for l in range(new_batch_size) ], dim=1)
			init_state = (init_h, init_c)
			enc_output2, enc_mask2 = [], []
			latent_var2 = []
			for batch_idx, left_size in enumerate(num_leftBeam):
				for beam_idx in range(left_size):
					enc_output2.append(enc_output[batch_idx].unsqueeze(0)) # (1, T, H)
					enc_mask2.append(enc_mask[batch_idx].unsqueeze(0)) # (1, T)
					latent_var2.append(latent_var[batch_idx].unsqueeze(0)) # (1, H)
			enc_output2 = torch.cat(enc_output2, dim=0)
			enc_mask2 = torch.cat(enc_mask2, dim=0)
			latent_var2 = torch.cat(latent_var2, dim=0)

			# get top k
			output, state = self._step(input_emb, init_state, latent_var2, enc_output2, enc_mask2, None, step_t, source) # (B, V), (L, B, H)
			top_value, top_index = torch.topk(torch.softmax(output, dim=1), k=beam_size, dim=1) # (B, K)
			state = (state[0].unsqueeze(1), state[1].unsqueeze(1)) # batch unsqueeze to speed up

			top_value = top_value.cpu() #.numpy() # use cpu is faster for this samll tensor (B, beam_size)
			top_value = torch.log(top_value) # take batch log to speed up
			top_index = top_index.cpu().numpy()
			for batch_idx, left_size in enumerate(num_leftBeam):
				start_idx = sum(num_leftBeam[:batch_idx])
				for beam_idx in range(left_size):
					beam = trim_pool[start_idx+beam_idx]
					# take shared logprob/state for candidate beams to speed up
					beam_logprob = top_value[start_idx+beam_idx]
					beam_state = (state[0][:, :, start_idx+beam_idx, :], state[1][:, :, start_idx+beam_idx, :])
					for cand_idx in range(beam_size):
						cand_beam = Beam(beam, None)

						# NOTE: avoiding using item() but converting to numpy() reduces half of decoding time
						cand_beam.history.append( self.idx2word[top_index[start_idx+beam_idx][cand_idx]] )

						# NOTE: be careful when updating logprob, DO NOT change logprob at previous step
						cand_beam.logprob += beam_logprob[cand_idx]
						cand_beam.state = beam_state
						cand_pool[batch_idx].append(cand_beam)
						logprob_pool[batch_idx].append(cand_beam.logprob)

			# sort candidate beams for each example AND update best beam so far
			global_pool = []
			for batch_idx in range(batch_size):
				if len(cand_pool[batch_idx]) == beam_size:
					global_pool.append(cand_pool[batch_idx])
					continue

				norm = torch.tensor([pow(len(beam.history)-1, alpha) for beam in cand_pool[batch_idx]]).float()
				_, indexes = torch.sort(torch.tensor(logprob_pool[batch_idx]).float()/norm,descending=True)
				cand_pool[batch_idx] = [cand_pool[batch_idx][idx] for idx in indexes.numpy()]
				global_pool.append(cand_pool[batch_idx][:beam_size])

		# re-format outputs
		sentences_batch, sentences_len_batch, logprobs_batch = [], [], torch.zeros(batch_size, beam_size)
		for batch_idx in range(batch_size):
			sentences, sentences_len, logprobs = [], [], []
			for beam_idx in range(beam_size):
				sentence_len = len(global_pool[batch_idx][beam_idx].history)-1 # miuns 1 for <SOS>
				global_pool[batch_idx][beam_idx].history.remove('<SOS>')
				if sentence_len < self.dec_len: global_pool[batch_idx][beam_idx].history.remove('<EOS>')
				sentence = ' '.join(global_pool[batch_idx][beam_idx].history)
				sentences.append(sentence)
				sentences_len.append(sentence_len)
				logprobs_batch[batch_idx][beam_idx] = global_pool[batch_idx][beam_idx].logprob / sentence_len
			sentences_batch.append(sentences)
			sentences_len_batch.append(sentences_len)

		# NOTE: DO NOT create new tensor as it does not have grad_fn
		logprobs_batch = logprobs_batch.cuda()
		return {'decode': sentences_batch, 'logprobs': logprobs_batch, 'decode_len': sentences_len_batch, 'mode': 'gen'}
