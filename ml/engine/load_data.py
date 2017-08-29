#coding=utf-8

def load_svm(input_file):
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


if __name__ == '__main__':
    labels, instances = load_svm('train.svm')
    print len(labels), len(instances)
    for i in range(len(labels)):
        print labels[i], instances[i]
   
