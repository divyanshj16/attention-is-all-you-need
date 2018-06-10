import pickle
import os
from process_data import f_de,f_en
import spacy
import numpy as np
import re

source_language = 'en'
target_language = 'de'
source_train = f_en
target_train = f_de
min_frequency = 15
en_vocab = './vocab/en.vocab'
de_vocab = './vocab/de.vocab'
max_len = 20
target_test_fname = './dataset/de-en/IWSLT16.TEDX.tst2014.de-en.de.xml'
source_test_fname = './dataset/de-en/IWSLT16.TEDX.tst2014.de-en.en.xml'


def create_vocab(ln,save=False):

	if ln == 'en':
		vocab_en = []
		for line in open(en_vocab):
			if int(line.split('\t')[1]) >= min_frequency:
				vocab_en.append(line.split('\t')[0])
			else: break
		word2idx_en = {word:i for i,word in enumerate(vocab_en)}
		idx2word_en = {i:word for i,word in enumerate(vocab_en)}
		if save:
			if not os.path.exists('mappings'): os.mkdir('mappings')
			pickle.dump(word2idx_en, open('mappings/word2idx_en.pkl','wb'))
			pickle.dump(idx2word_en,open('mappings/idx2word_en.pkl','wb'))
		return word2idx_en, idx2word_en

	if ln == 'de':
		vocab_de = []
		for line in open(de_vocab):
			if int(line.split('\t')[1]) >= min_frequency:
				vocab_de.append(line.split('\t')[0])
			else: break
		word2idx_de = {word:i for i,word in enumerate(vocab_de)}
		idx2word_de = {i:word for i,word in enumerate(vocab_de)}
		if save:
			if not os.path.exists('mappings'): os.mkdir('mappings')
			pickle.dump(word2idx_de,open('mappings/word2idx_de.pkl','wb'))
			pickle.dump(idx2word_de,open('mappings/idx2word_de.pkl','wb'))
		return word2idx_de, idx2word_de

def create_train_data(save=False):
	source_sentences = [line for line in open(source_train,'r').read().split('\n') if len(line) != 0 and line[0] != '<']
	target_sentences = [line for line in open(target_train,'r').read().split('\n') if len(line) != 0 and line[0] != '<']

	ss_arr,ts_arr, _, _ = create_data(source_sentences,target_sentences)
	# ss_arr, ts_arr = np.array(ss_arr), np.array(ts_arr)

	if save == True:
		np.save('mappings/ss_arr_train.npy',ss_arr)
		np.save('mappings/ts_arr_train.npy',ts_arr)

	return ss_arr,ts_arr


def create_data(source_sentences, target_sentences):
	print('Creating data.....')
	tar2idx, idx2tar = create_vocab(target_language, save=True)
	sour2idx, idx2sour = create_vocab(source_language, save=True)

	source_tokenizer = spacy.load(source_language).tokenizer
	target_tokenizer = spacy.load(target_language).tokenizer

	x_all , y_all, source_all, tar_all = [], [] , [], []
	for sour_sen, tar_sen in zip(source_sentences,target_sentences):
		x = [sour2idx.get(str(word),1) for word in list(source_tokenizer(sour_sen))]
		y = [tar2idx.get(str(word),1) for word in list(target_tokenizer(tar_sen))]
		x += [3] # adding </s> character to show end of line
		y += [3]
		if max(len(x), len(y)) <= max_len:
			x += [0] * (max_len - len(x))
			y += [0] * (max_len - len(y))
			x_all.append(x)
			y_all.append(y)
			source_all.append(source_sentences)
			tar_all.append(target_sentences)		
		# print(len(x),len(y))

		# print(x,sour_sen)
		# print(y,tar_sen)
		# break
	x_all = np.array(x_all)
	y_all = np.array(y_all)

	return x_all,y_all,source_all,tar_all

def create_test_data(name_suff='test',save=False):
	def _refine(line):
		line = re.sub("<[^>]+>", "", line)
		return line
	
	en_sents = [_refine(line) for line in open(source_test_fname, 'r').read().split("\n") if line and line[:4] == "<seg"]
	de_sents = [_refine(line) for line in open(target_test_fname, 'r').read().split("\n") if line and line[:4] == "<seg"]
		
	ss_test, ts_test, sources_test, targets_test = create_data(en_sents, de_sents)

	if save == True:
		np.save(f'mappings/ss_arr_{name_suff}.npy',ss_test)
		np.save(f'mappings/ts_arr_{name_suff}.npy',ts_test)
	return ss_test, ts_test, sources_test, targets_test


if __name__ == '__main__':
	# print(create_vocab(target_language)[0])
	create_train_data(save=True)
	create_test_data(save=True)