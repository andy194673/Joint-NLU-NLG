import sys
import nltk
from nltk.translate.bleu_score import sentence_bleu

def e2e_metric(decodes, refs, config, beam_search, dType, dataset):
	'''
	Args:
		decodeds: a dict where key is query/parse and value is a list of hypotheses
		refs: a dict where key is query/parse and value is a list of references
	Return:
		result of source. bleu for query generation and match accuracy for parsing
	'''
	def parse_mr(mr):
		slot2value = {}
		sv_list = []
		toks = mr.split(' | ')
		for tok in toks:
			slot = tok.split()[0]
			value = ' '.join(tok.split()[1:])
			slot2value[slot] = value
			sv_list.append(slot+'-'+value)
		return slot2value, sv_list

	def compute_f1(mr_hyp, mr_ref):
		mr_hyp, list_hyp = parse_mr(mr_hyp) # slot2value dict & sv pair list
		mr_ref, list_ref = parse_mr(mr_ref)

		# slot f1
		TP = 0
		for slot in mr_hyp:
			if slot in mr_ref:
				TP += 1
		prec = TP / len(mr_hyp)
		rec  = TP / len(mr_ref)
		try:
			slot_f1 = 2*prec*rec/(prec+rec)
		except ZeroDivisionError:
			slot_f1 = 0

		# value f1
		TP = 0
		for sv in list_hyp:
			if sv in list_ref:
				TP += 1
		prec = TP / len(list_hyp)
		rec  = TP / len(list_ref)
		try:
			value_f1 = 2*prec*rec/(prec+rec)
		except ZeroDivisionError:
			value_f1 = 0
		return slot_f1, value_f1

	def delex(string):
		for slot, values in dataset.ontology.items():
			if slot == 'family_friendly':
				continue
			for value in values:
				if value in string:
					string = string.replace(value, '__'+slot+'__')
		return string

	sources = config.dec_side
	assert len(refs['query']) == len(refs['parse'])
	if 'parse' in sources:
		assert len(refs['parse']) == len(decodes['parse'])
	if 'query' in sources:
		assert len(refs['query']) == len(decodes['query'])

	output = []
	slot_f1, value_f1, joint_acc = 0, 0, 0

	# build multiple references for calulating bleu
	parse2refs = {}
	for exampleIdx, (query_ref, parse_ref) in enumerate(zip(refs['query'], refs['parse'])):
		query_ref_delex = delex(query_ref)
		parse_ref_delex = delex(parse_ref)
		if parse_ref_delex not in parse2refs:
			parse2refs[parse_ref_delex] = set()
		parse2refs[parse_ref_delex].add(query_ref) # store lexicalised string

	# iterate each example in test set
	value_count, value_match = 0, 0
	sent_bleu = 0
	for exampleIdx, (query_ref, parse_ref) in enumerate(zip(refs['query'], refs['parse'])):
		parse_hyp = decodes['parse'][exampleIdx]
		assert isinstance(parse_hyp, str)
		_slot_f1, _value_f1 = compute_f1(parse_hyp, parse_ref)
		slot_f1 += _slot_f1
		value_f1 += _value_f1
		joint_acc += parse_hyp==parse_ref

		query_hyp = decodes['query'][exampleIdx]
		assert isinstance(query_hyp, str)
		value_match_turn = 0
		mr_ref, _ = parse_mr(parse_ref) # slot2value dict & sv pair list
		for slot, value in mr_ref.items():
			if value in ['no', 'yes']:
				value_match_turn += 1
				continue
			value_count += 1
			if value in query_hyp:
				value_match += 1
				value_match_turn += 1

		output.append(['PARSE MATCH: '+str(parse_ref==parse_hyp), 'PARSE REF: '+parse_ref, 'PARSE HYP: ' +parse_hyp, \
						'QUERY MATCH: '+str(value_match_turn==len(mr_ref)), 'QUERY REF: '+query_ref, 'QUERY HYP: ' +query_hyp])

		# calculate bleu using NLTK toolkit
		references = [ref.split() for ref in list(parse2refs[delex(parse_ref)])]
		sent_bleu += sentence_bleu(references, query_hyp.split()) # NOTE: use delex or not does not really matter for calculating bleu, bleu_lex

	# avg over examples
	sent_bleu /= len(refs['query'])
	if 'parse' in sources:
		slot_f1 /= len(refs['parse'])
		value_f1 /= len(refs['parse'])
		joint_acc /= len(refs['parse'])
	else:
		parse_acc = 0

	if 'query' in sources:
		value_str_match_acc = value_match / value_count
	else:
		bleu, bleu_group = 0, 0

	return slot_f1, value_f1*100, joint_acc*100, sent_bleu*100, value_str_match_acc*100, output
	
	

def weather_metric(decodes, refs, config, beam_search, dType):
	'''
	Args:
		source: list of source, e.g., ['query', 'parse']
		decodeds: a dict where key is query/parse and value is a list of hypotheses (str)
		refs: a dict where key is query/parse and value is a list of references (str)
	Return:
		result of source. bleu for query generation and match accuracy for parsing
	'''
	sources = config.dec_side
	assert len(refs['query']) == len(refs['parse'])
	if 'parse' in sources:
		assert len(refs['parse']) == len(decodes['parse'])
	if 'query' in sources:
		assert len(refs['query']) == len(decodes['query'])

	output = []
	parse_acc = 0
	parse2refs = {}
	unique_query = set([])
	for exampleIdx, (query_ref, parse_ref) in enumerate(zip(refs['query'], refs['parse'])):
		if parse_ref not in parse2refs:
			parse2refs[parse_ref] = set()
		parse2refs[parse_ref].add(query_ref)
		unique_query.add(query_ref)

	c = 0
	for parse, REFS in parse2refs.items():
		c += len(REFS)

	sent_bleu = 0
	for exampleIdx, (query_ref, parse_ref) in enumerate(zip(refs['query'], refs['parse'])):
		parse_hyp = decodes['parse'][exampleIdx]
		assert isinstance(parse_hyp, str)
		if parse_hyp == parse_ref:
			parse_acc += 1
		query_hyp = decodes['query'][exampleIdx]
		assert isinstance(query_hyp, str)
		output.append(['PARSE MATCH: '+str(parse_ref==parse_hyp), 'PARSE REF: '+parse_ref, 'PARSE HYP: ' +parse_hyp, \
				'QUERY MATCH: '+str(query_ref==query_hyp), 'QUERY REF: '+query_ref, 'QUERY HYP: ' +query_hyp])
		references = [ref.split() for ref in list(parse2refs[parse_ref])]
		sent_bleu += sentence_bleu(references, query_hyp.split())

	sent_bleu /= len(refs['query'])
	if 'parse' in sources:
		parse_acc /= len(refs['parse'])
	else:
		parse_acc = 0
	return parse_acc*100, sent_bleu*100, sent_bleu*100, output
