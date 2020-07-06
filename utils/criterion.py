import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

# class/function for calculating loss is adapted from the following repo
# https://github.com/snakeztc/NeuralDialog-LaRL/blob/master/latent_dialog/criterions.py

def NLLEntropy(logits, target, ignore_idx=None):
	'''
	Args:
		logits: (B, T, V)
		target: (B, T)
	'''
	B, T, V = logits.size()
	logProb = F.log_softmax(logits.view(B*T, V), dim=1) 
	loss = F.nll_loss(logProb, target.view(-1), reduction='mean', ignore_index=ignore_idx)
	return loss


class NormKLLoss(_Loss):
	def __init__(self, unit_average=False):
		super(NormKLLoss, self).__init__()
		self.unit_average = unit_average

	def forward(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
		'''
		Args:
			recog_mu, recog_lovar, prior_mu, prior_logvar: (B, Z)
		Return:
			kl_loss: (B,)
		'''
		loss = 1.0 + (recog_logvar - prior_logvar)
		loss -= torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
		loss -= torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar))
		if self.unit_average:
			kl_loss = -0.5 * torch.mean(loss, dim=1)
		else:
			kl_loss = -0.5 * torch.sum(loss, dim=1)
		return torch.mean(kl_loss)

	def merge(self, kl1, kl2, source):
		'''
		Merge two kl losses kl(p|q), kl(q|p) based on its source of sampling 
		Args:
			kl1, kl2: (B,)
			source: a list (len=B)
		Return:
			average kl loss
		'''
		batch_size = kl1.size(0)
		assert batch_size == len(source)
		kl = torch.zeros(batch_size).cuda()
		for idx, s in enumerate(source):
			if s == 0:
				kl[idx] = kl1[idx]
			elif s == 1:
				kl[idx] = kl2[idx]
			else:
				raise ValueError('Unknown source of sampling')
		return torch.mean(kl)
