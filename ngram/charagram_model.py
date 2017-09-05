#encoding=utf8
import numpy as np
import charagram_util as cutil

def load_embedding_file(infile, vocab_file, dict_threshold):
    vec_size = len(open(infile,'r').readline().strip().split('\t')[1].split(','))
    vocab_map = cutil.load_dict(vocab_file, dict_threshold)
    print 'init_embedding_file, vec_size = %s' % vec_size
    init_embedding = np.zeros((len(vocab_map)+2, vec_size))
    not_find_count = 0
    for line in open(infile,'r'):
        infos = line.strip().split('\t')
        if infos[0] in vocab_map:
            init_embedding[vocab_map[infos[0]]] = np.asarray( map(float,infos[1].split(',')) )
        else:
            not_find_count += 1
    print 'load_embedding_file, not_find_count = %s' % not_find_count
    return init_embedding

def save_embedding_file(embedding_wight, vocab_file, out_file):
    vocab_map = {v:k for k,v in cutil.load_dict(vocab_file).items()}
    fout = open(out_file,'w')
    for i in range(len(embedding_wight[0])):
        if i in vocab_map:
            fout.write(vocab_map[i] + '\t' + ','.join(map(str,embedding_wight[0][i])) + '\n')
    fout.close()
