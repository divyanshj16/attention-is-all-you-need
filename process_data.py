import spacy
from collections import Counter
import os

f_en = './dataset/de-en/train.tags.de-en.en'
f_de = './dataset/de-en/train.tags.de-en.de'

def create_vocab_from_file(inp,op,ln):
    nlp = spacy.load(ln)
    with open(inp) as f:
        text = f.read()
        doc = nlp.tokenizer(text)
        doc = [str(i) for i in list(doc)]
        cntr = Counter(doc)
        if not os.path.exists('vocab'): os.mkdir('vocab')
        with open(f'vocab/{op}','w') as fo:
            fo.write("{}\t1000\n{}\t1000\n{}\t1000\n{}\t1000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
            for word, cnt in cntr.most_common(len(cntr)):
                fo.write(f'{word}\t{cnt}\n')
                
if __name__ == '__main__':
    create_vocab_from_file(f_en,'en.vocab','en')
    # create_vocab_from_file(f_de,'de.vocab','de')
    print("Vocab Created!!")
    