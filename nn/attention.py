import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class Attn(nn.Module):
	def __init__(self, context_size, query_size, method='cat'):
		super(Attn, self).__init__()
		self.method = method
		hidden_size = query_size
		self.H = hidden_size

		if self.method == 'cat':
			self.attn = nn.Linear(context_size + query_size, hidden_size)
			self.v = nn.Linear(hidden_size, 1, bias=False)

#		elif self.method == 'dot':
#			self.attn = nn.Linear(self.hidden_size, hidden_size)

		else:
			print('Wrong type of attention mechanism', file=sys.stderr)
			sys.exit(1)


	def forward(self, query, context, mask, return_energy=False):
		'''
		Args:
			query: decoder hidden state (B, 1, H)
			context: encoder hidden states (B, T, H)
			mask: encoder len (B, T)
		Returns:
			context_vec: context vector (B, H)
			attention dist (B, T)
		'''
		self.B, self.T, _ = context.size()
		query = query.expand(self.B, self.T, self.H)
		attn_dist = self.score(query, context, mask) # (B, T)
		context_vec = torch.bmm(attn_dist.unsqueeze(1), context) # (B, 1, H)
		return attn_dist, context_vec.squeeze(1)


	def score(self, query, context, mask):
		if self.method == 'cat':
			score = torch.tanh(self.attn(torch.cat([query, context], 2))) # (B, T, feat_size) => (B, T, H)
			score = self.v(score) # (B, T, 1)
			score = (score.squeeze(2) + mask).view(self.B, self.T) # (B, T)
		return torch.softmax(score, dim=1)
