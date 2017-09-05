#coding=utf-8
import sys
import numpy as np

def load_svm(input_file): #sparse
    labels, instances = [], []
    pos, neg = 0, 0
    for line in open(input_file):
        tokens = line.strip().split()
        label = int(tokens[0])
        if label == -1:
            label = 0
        inst = {}
        for token in tokens[1:]:
            k,v = token.split(':')
            inst[k] = float(v)
        if inst:
            labels.append(label)
            instances.append(inst)
            if label > 0.5:
                pos += 1
            else:
                neg += 1
    print input_file, 'neg:', neg, 'pos:', pos, 'pos ratio:', float(pos)/(neg+pos)
    return labels, instances

def coarse_static(input_file):
    insts_num, max_feature = 0, 0
    pos_num, neg_num = 0, 0
    for line in open(input_file):
        insts_num += 1
        tokens = line.strip().split()
        if float(tokens[0]) > 0.5:
            pos_num += 1
        else:
            neg_num += 1
        max_tmp = max([int(token.split(':')[0]) for token in tokens])
        max_feature = max(max_tmp, max_feature)
    return {'insts_num':insts_num, 'max_feature':max_feature, 'pos_num':pos_num, 'neg_num':neg_num}

def load_svm_as_numpy_array(input_file, max_feature):
    stats = coarse_static(input_file)
    if max_feature and max_feature < stats['max_feature']:
        print 'error: arg max_feature < real max_feature'
        sys.exit(0)
    if not max_feature:
        max_feature = stats['max_feature']
    labels = np.zeros(stats['insts_num'], dtype=int)
    insts = np.zeros((stats['insts_num'], max_feature), dtype=float)
    row = 0
    for line in open(input_file):
        tokens = line.strip().split()
        labels[row] = int(tokens[0])
        for token in tokens[1:]:
            k,v = token.split(':')
            insts[row][int(k)-1] = float(v)
        row += 1
    return labels, insts, max_feature

if __name__ == '__main__':
    labels, insts = load_svm(sys.argv[1])
    print len(labels), len(insts)

    stats = {}
    labels, insts, max_feature = load_svm_as_numpy_array(sys.argv[1], None)
    print len(labels), len(insts), max_feature
