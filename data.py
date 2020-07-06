import json
import sys
import random
import torch

class DataLoader():
	def __init__(self, config):
		self.config = config
		self.build_vocab()

		# collect data
		self.data = {'train': [], 'valid': [], 'test': [], 'unsup_parse': [], 'unsup_query': []}
		self.parseData(config.train_path, 'train')
		self.parseData(config.valid_path, 'valid')
		self.parseData(config.test_path, 'test')

		if config.unsup_learn:
			self.parseData(config.unsup_parse_path, 'unsup_parse')
			self.parseData(config.unsup_query_path, 'unsup_query')

		self.idx2word = {}
		self.idx2word['query'] = {idx: w for w, idx in self.vocab['query'].items()}
		self.idx2word['parse'] = {idx: w for w, idx in self.vocab['parse'].items()}

		if config.dataset == 'e2e':
			self.loadOntology()
			self.slot_list = ['name', 'eat_type', 'food', 'price_range', 'customer_rating', 'area', 'family_friendly', 'near']
		self.init()

	def loadOntology(self):
		with open(self.config.ontology) as f:
			self.ontology = json.load(f)

	def build_vocab(self):
		self.vocab = {}
		self.vocab['query'] = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3} # word2id for query
		with open(self.config.word2count_query) as f:
			word2count = json.load(f)
		for i in range(min(self.config.vocab_size, len(word2count))):
			w = word2count[i][0]
			self.vocab['query'][w] = len(self.vocab['query'])

		self.vocab['parse'] = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3} # word2id for parse
		with open(self.config.word2count_parse) as f:
			word2count = json.load(f)
		for i in range(min(self.config.vocab_size, len(word2count))):
			w = word2count[i][0]
			self.vocab['parse'][w] = len(self.vocab['parse'])


	def next_batch(self, dType):
		if self.p[dType] >= len(self.data[dType]):
			if dType in ['train', 'valid', 'test']:
				return None
			else:
				self.p[dType] %= len(self.data[dType])

		x_idx, y_idx = [], []
		x_ref, y_ref = [], []
		p_start = self.p[dType]
		B = 0
		while B < self.config.batch_size and self.p[dType] < len(self.data[dType]):
			if self.data[dType][self.p[dType]]['query'] is not None:
				x_idx.append(list(self.data[dType][self.p[dType]]['query'])) # NOTE: create new list here, DO NOT CHANGE ORIGINAL DATA
			else:
				x_idx.append(None)
			if self.data[dType][self.p[dType]]['parse'] is not None:
				y_idx.append(list(self.data[dType][self.p[dType]]['parse']))
			else:
				y_idx.append(None)
			x_ref.append(self.data[dType][self.p[dType]]['ref-query'])
			y_ref.append(self.data[dType][self.p[dType]]['ref-parse'])
			B += 1
			self.p[dType] += 1

		if x_idx[0] is not None:
			# padding
			x_len = [len(sent) for sent in x_idx]
			max_len = max(x_len)
			for sent in x_idx: sent.extend( [self.vocab['query']['<PAD>'] for _ in range(max_len-len(sent))] )

			# mask on x side
			x_mask = []
			max_len = max(x_len)
			for s_len in x_len:
				mask = [0 for _ in range(s_len)] + [float('-inf') for _ in range(max_len-s_len)]
				x_mask.append(mask)

		if y_idx[0] is not None:
			# padding
			y_len = [len(sent) for sent in y_idx]
			max_len = max(y_len)
			for sent in y_idx: sent.extend( [self.vocab['parse']['<PAD>'] for _ in range(max_len-len(sent))] )

			# mask on y side
			y_mask = []
			max_len = max(y_len)
			for s_len in y_len:
				mask = [0 for _ in range(s_len)] + [float('-inf') for _ in range(max_len-s_len)]
				y_mask.append(mask)

		# nlu classification target
		if self.config.dataset == 'e2e':
			target = {}
			for slot in self.slot_list:
				target[slot] = []
			for b, ref in enumerate(y_ref):
				s2v = {}
				for sv in ref.split(' | '):
					slot = sv.split()[0]
					value = ' '.join(sv.split()[1:])
					s2v[slot] = value
	
				for slot in self.slot_list:
					if slot in s2v:
						value = s2v[slot]
						value_idx = self.ontology[slot].index(value)
						target[slot].append(value_idx)
					else:
						target[slot].append(len(self.ontology[slot]))
			for slot in self.slot_list:
				target[slot] = torch.tensor(target[slot]).long().cuda()

		# return tensor
		batch = {'word_idx': {}, 'sent_len': {}, 'mask': {}, 'ref': {}, 'flag': {}, 'class': {}}
		if x_idx[0] is not None: # x is given
			batch['word_idx']['query'] = torch.tensor(x_idx).long().cuda()
			batch['sent_len']['query'] = torch.tensor(x_len).long().cuda()
			batch['mask']['query'] = torch.tensor(x_mask).float().cuda()
			batch['ref']['query'] = x_ref
		else:
			batch['word_idx']['query'] = None
			batch['sent_len']['query'] = None
			batch['mask']['query'] = None
#			batch['ref']['query'] = None
			batch['ref']['query'] = [self.data['unsup_query'][i]['ref-query'] for i in range(p_start, self.p[dType])] # TODO back

		if y_idx[0] is not None: # y is given
			batch['word_idx']['parse'] = torch.tensor(y_idx).long().cuda()
			batch['sent_len']['parse'] = torch.tensor(y_len).long().cuda()
			batch['mask']['parse'] = torch.tensor(y_mask).float().cuda()
			batch['ref']['parse'] = y_ref
#			batch['flag']['parse'] = torch.tensor(y_flag).float().cuda() # BACK
			if self.config.dataset == 'e2e':
				batch['class']['parse'] = target
		else:
			batch['word_idx']['parse'] = None
			batch['sent_len']['parse'] = None
			batch['mask']['parse'] = None
#			batch['ref']['parse'] = None
			batch['ref']['parse'] = [self.data['unsup_parse'][i]['ref-parse'] for i in range(p_start, self.p[dType])] # TODO back

		if dType == 'train':
			batch['run_auto'] = {'query': True, 'parse': True}
		else:
			batch['run_auto'] = {'query': False, 'parse': False}
		return batch
		

	def init(self):
		self.p = {'train': 0, 'valid': 0, 'test': 0, 'unsup_query': 0, 'unsup_parse': 0}
		if self.config.shuffle:
			random.shuffle(self.data['train'])
			if self.config.unsup_learn:
				unsup = list(zip(self.data['unsup_query'], self.data['unsup_parse']))
				random.shuffle(unsup)
				self.data['unsup_query'], self.data['unsup_parse'] = zip(*unsup)


	def parseData(self, path, dType):
		f = open(path)
		for line in f.readlines():
			example = {}
			query, parse = line.strip().split('\t')
			if query != 'none':
				query_idx = self.parseSent(query, self.vocab['query'])
				example['ref-query'] = query.strip()
				example['query'] = query_idx
			else:
				example['ref-query'] = query
				example['query'] = None

			if parse != 'none':
				parse_idx = self.parseSent(parse, self.vocab['parse'])
				example['ref-parse'] = parse.strip()
				example['parse'] = parse_idx
			else:
				example['ref-parse'] = parse
				example['parse'] = None
			self.data[dType].append(example)


	def parseSent(self, sent, vocab):
		sent_idx = []
		for tok in sent.split():
			tok_idx = vocab[tok] if tok in vocab else vocab['<UNK>']
			sent_idx.append(tok_idx)
		sent_idx.append(vocab['<EOS>']) # add eos at the end of sentence
		return sent_idx


if __name__ == '__main__':
	pass
