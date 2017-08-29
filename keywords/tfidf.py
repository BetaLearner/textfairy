#coding=utf8
import sys, os, multiprocessing
import time, math
import mr.mr as mr

class TfIdfMapper(mr.Mapper):
    def __init__(self, rank, file_list, output_dir, params):
        super(TfIdfMapper, self).__init__(rank, file_list, output_dir, params)
        self.word_id = {}
        self.id_idf = {}
        self.load_idf(self.params['idf_file'])
    
    def load_idf(self, idf_file):
        doc_cnt = 0
        doc_cnt_init = False
        for line in open(idf_file):
            tokens = line.strip().split()
            if 1 == len(tokens) and not doc_cnt_init:
                doc_cnt = int(tokens[0])
                doc_cnt_init = True
            elif 3 == len(tokens):
                self.word_id[tokens[1]] = tokens[0]
                self.id_idf[tokens[0]] = math.log( (doc_cnt-int(tokens[2])+0.5) / (int(tokens[2])+0.5) )
            else:
                print 'load_idf, mapper:%s input error, line:%s' % (rank, line)
                sys.exit(0)

    def process(self, line):
        tokens = line.strip().split()
        word_tf = {}
        doc_len = float(len(tokens) - 1)
        for token in set(tokens[1:]):
            word_tf[token] = word_tf.get(token,0) + 1
        tfidfs = []
        for word in word_tf:
            if word in self.word_id:
                wid = self.word_id[word]
                tfidfs.append('%s:%6f' % (wid, word_tf[word]/doc_len * self.id_idf[wid]))
        self.output.write(tokens[0] + ' ' + ' '.join(tfidfs) + '\n')

    def done(self):
        self.output.close()

