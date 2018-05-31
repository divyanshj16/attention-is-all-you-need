import pickle
import os

min_frequency = 15
en_vocab = './vocab/en.vocab'
de_vocab = './vocab/de.vocab'


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
			pickle.dump(word2idx_en,'mappings/word2idx_en')
			pickle.dump(idx2word_en,'mappings/idx2word_en')
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
			pickle.dump(word2idx_de,'mappings/word2idx_de')
			pickle.dump(idx2word_de,'mappings/idx2word_de')
		return word2idx_de, idx2word_de

def create_train_data():
	pass

def create_data(source_sentences, target_sentences):
	de2idx, idx2de = create_vocab('de')
	en2idx, idx2en = create_vocab('en')

	x , y, source, tar = [], [] , [], []
	for sour_sen, tar_sen in zip(source_sentences,target_sentences):
		x = [de2idx.get(word,1) for word in sour_sen.split()]
		y = [en2idx.get(word,1) for word in tar_sen.split()]



if __name__ == '__main__':
	print(create_vocab('de')[0])