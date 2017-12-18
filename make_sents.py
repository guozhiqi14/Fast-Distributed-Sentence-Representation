import codecs
import numpy as np
import cPickle as pickle

vocabs = ['</S>', '<S>'] + list(map(lambda x: x.strip(), open('SortedVocab.txt').readlines()))
rev_voc_encode = dict([(w, i) for i, w in enumerate(vocabs[:200000])])
sens = []
f = codecs.open('books_large_p1.txt', 'r', encoding='iso-8859-1')
k = 10000
for line in f:
    sens.append(line.strip().encode('ascii', 'replace'))
    if k==60000:
    	break
    k+=1
f.close()
sens_split = [['<S>'] + ex.split() for ex in sens]
max_l = 64
sen_trunc = [sen[:max_l - 1] for sen in sens_split]
sens_num = np.array([[rev_voc_encode.get(w, 3) for w in sen] for sen in sen_trunc]) 
pickle.dump(sens_num, open('sens_for_wilson_offset.pk', 'wb'))
