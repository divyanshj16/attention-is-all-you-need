import spacy

ent = spacy.load('en')
# def make_vocab():


file_name = './dataset/de-en/train.tags.de-en.en'

with open(file_name) as f:
	text = f.read()
	temp = text[500:1000]
	doc = ent(temp)
	print(doc.to_array(doc.vocab))
	# for w in doc: print(w.text,w.pos_)
	# print(doc)

	# new_text = regex.sub("[^\s\p{Latin}']", "", text)
