#encoding=utf8
import numpy as np
import random, sys

def negativeSampling(data_t, data_c, bias, pos_size):
    sample_t_idx = [] 
    sample_c_idx = []
    labels = np.zeros(pos_size*2, dtype='bool')
    for i in range(pos_size):
        sample_t_idx.append((bias+i) % len(data_t))
        sample_c_idx.append((bias+i) % len(data_c))
        labels[i] = True
    for i in range(pos_size):
        r1 = random.randint(0, pos_size)
        r2 = random.randint(0, pos_size)
        if r1 == r2:
            r2 = (r1+1) % pos_size
        sample_t_idx.append((bias+r1) % len(data_t))
        sample_c_idx.append((bias+r2) % len(data_c))
        labels[pos_size+i] = False
    p = np.random.permutation(len(sample_t_idx))
    sample_t_idx = [sample_t_idx[i] for i in p]
    sample_c_idx = [sample_c_idx[i] for i in p]
    labels = np.array([labels[i] for i in p])
    return data_t[sample_t_idx], data_c[sample_c_idx], labels
    

