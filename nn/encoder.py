import torch
import torch.nn as nn

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5, bidirectional=True):
		super(RNN, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, \
				dropout=dropout, bidirectional=bidirectional, batch_first=True)

		self.num_di = 2 if bidirectional else 1
		self.hidden_size = hidden_size
		self.num_layers = num_layers


	def forward(self, input_var, input_len, init_state=None):
		'''
		input_var: (B, T, input_size)
		input_len: (B)
		init_state: (num_layers * num_directions, B, H)
		'''
		batch_size, _, _ = input_var.size()
		# NOTE: set enforce_sorted True to avoid sorting
		input_var = nn.utils.rnn.pack_padded_sequence(input_var, input_len, batch_first=True, enforce_sorted=False)

		# output shape: (batch, seq_len, num_directions * hidden_size)
		# state (h, c) shape: (num_layers * num_directions, batch, hidden_size)
		if init_state is not None:
			output, state = self.rnn(input_var, init_state)
		else:
			output, state = self.rnn(input_var)
		
		output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

		if isinstance(state, tuple):
			h, c = state
			h = h.transpose(0, 1).contiguous().view(self.num_layers, batch_size, self.num_di*self.hidden_size)
			c = c.transpose(0, 1).contiguous().view(self.num_layers, batch_size, self.num_di*self.hidden_size)
			state = (h, c)
		else:
			state = state.transpose(0, 1).contiguouse().view(batch_size, self.di*self.hidden_size)
		
		return output, state


class SentEncoder(nn.Module):
	def __init__(self, input_size, embed_size, hidden_size, num_layers=1, dropout=0.5, bidirectional=True):
		super(SentEncoder, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.embed = nn.Embedding(input_size, embed_size)
		self.rnn = RNN(embed_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

	def forward(self, input_var, input_len, init_state=None):
		'''
		Args:
			input_var: (B, T)
		'''
		embed = self.embed(input_var) # (B, T, E)
		embed = self.dropout(embed)
		output, state = self.rnn(embed, input_len, init_state) # (B, T, num_di*H) & (num_layer, B, num_di*H)
		return output, state
