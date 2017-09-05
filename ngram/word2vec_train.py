#encoding=utf8
import gensim, sys
from gensim.models.word2vec import Word2Vec

fout = open('word2vec_train.dat','w')
for line in open(sys.argv[1],'r'):
    infos = line.strip().split('\t')
    if len(infos) != 3:
        continue
    fout.write(infos[1]+'\n')
    fout.write(infos[2]+'\n')
fout.close()
    
w2v = Word2Vec(gensim.models.word2vec.LineSentence('word2vec_train.dat'), size=64, iter=10)

for item in w2v.most_similar(u'水',topn=5):
    print item[0].encode('utf8')

fout = open('word2vec_64_iter10.embedding_test','w')
print len(w2v.vocab)
for word in w2v.vocab:
    fout.write(word.encode('utf8') + '\t' + ','.join(map(str,w2v[word])) + '\n')
fout.close()


