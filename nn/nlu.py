import sys
import torch
import torch.nn as nn

class NLU(nn.Module):
	'''
	A set of several classifiers, which is used as a decoder in NLU path for E2E dataset.
	'''
	def __init__(self, dataset, config, input_size):
		super(NLU, self).__init__()
		self.config = config
		self.ontology = dataset.ontology
		self.classifier = nn.ModuleDict({})

		if config.nlu_embed:
			self.embed = nn.Linear(input_size, config.embed_size)
		self.slot_list = dataset.slot_list

		for slot in self.slot_list: # ordered for evaluation
			value_list = self.ontology[slot]
			output_size = len(value_list) + 1 # 1 for 'none' value
			if config.nlu_embed:
				self.classifier[slot] = nn.Sequential(nn.Linear(config.embed_size, output_size))
			else:
				self.classifier[slot] = nn.Sequential(nn.Linear(input_size, output_size))

		self.criterion = []
		for slot in self.slot_list: # ordered for evaluation
			value_list = self.ontology[slot]
			n_class = len(value_list) + 1 # 1 for 'none' value
			weight = [1. for _ in range(n_class)]
			weight[-1] = self.config.none_weight
			self.criterion.append(nn.CrossEntropyLoss(weight=torch.tensor(weight).float().cuda()))
			# NOTE: use nn.CrossEntropyLoss for stability; avoid nan


	def forward(self, latent_var, beam_search=False):
		'''
			latent_var: (B, Z)
			if beam_search is True, all output distribution is repeated beam_search times, e.g., (B, C) -> (B'=B*beam_size, C)
			beam_search switch is meaningless for classification, but easier for implmentation.
		'''
		batch_size = latent_var.size(0)
		if self.config.nlu_embed:
			e = self.embed(latent_var) # (B, E)

		output_dist, output_seq = {}, [[] for _ in range(batch_size)]
		for slot in self.slot_list: # ordered for evaluation
			if self.config.nlu_embed:
				value_dist = self.classifier[slot](e) # (B, num_value)
			else:
				value_dist = self.classifier[slot](latent_var) # (B, num_value)

			if beam_search: # no beam search actually, this is just for re-shaping the tensor
				output_dist[slot] = value_dist.unsqueeze(1).repeat(1, self.config.beam_size, 1) \
					.view(batch_size*self.config.beam_size, -1) # (B', C) 
			else:
				output_dist[slot] = value_dist # (B, C)

			# collect output seq for evaluation
			prob, idx = torch.max(torch.softmax(value_dist, dim=1), dim=1) # (B,)
			value_list = self.ontology[slot]
			for b in range(batch_size):
				value_idx = idx[b].item()
				if value_idx < len(value_list): # if predicted value is not 'none'
					pred_value = value_list[value_idx]
					output_seq[b].append(slot)
					output_seq[b].append(pred_value)
					output_seq[b].append('|')

		decode_len = []
		for b in range(batch_size):
			seq = ' '.join(output_seq[b][:-1])
			if beam_search:
				output_seq[b] = [seq for _ in range(self.config.beam_size)]
				decode_len.append( [len(seq.split())+1 for _ in range(self.config.beam_size)] )
			else:
				output_seq[b] = seq
				decode_len.append(len(seq.split())+1) # +1 for <EOS>

		if beam_search:
			logprobs = []
			for slot in self.slot_list:
				value, index = torch.max(torch.softmax(output_dist[slot], dim=1), dim=1) # (B')
				logprobs.append(torch.log(value).unsqueeze(1))
			logprobs = torch.cat(logprobs, dim=1) # (B', #slots)
			logprobs = torch.mean(logprobs, dim=1) # (B')
			return {'decode': output_seq, 'output_dist': output_dist, 'mode': 'gen', \
						'logprobs': logprobs.view(batch_size, self.config.beam_size), 'decode_len': decode_len}
		else:
			return {'decode': output_seq, 'output_dist': output_dist, 'mode': 'teacher_force'}
		

	def get_loss(self, output_dist, target):
		# get loss per slot
		loss = 0
		for idx, slot in enumerate(self.slot_list):
			loss += self.criterion[idx](output_dist[slot], target[slot])
		loss /= len(self.slot_list)
		return loss
