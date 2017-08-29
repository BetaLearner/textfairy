#coding=utf8
import sys, os, multiprocessing
import time
import mr.mr as mr

class IdfMapper(mr.Mapper):
    def __init__(self, rank, file_list, output_dir, params):
        super(IdfMapper, self).__init__(rank, file_list, output_dir, params)
        self.doc_cnt = 0
        self.idf_cnt = {}

    def process(self, line):
        tokens = line.strip().split()
        for token in set(tokens[1:]):
            self.idf_cnt.setdefault(token, 0)
            self.idf_cnt[token] += 1
        self.doc_cnt += 1

    def done(self):
        self.output.write('%s\n' % self.doc_cnt)
        for token in self.idf_cnt:
            self.output.write('%s %s\n' %(token, self.idf_cnt[token]))
        self.output.close()

class IdfReducer(mr.Reducer):
    def __init__(self, file_list, output_dir, params, output_file=None):
        super(IdfReducer, self).__init__(file_list, output_dir, params, output_file)
        self.doc_cnt = 0
        self.idf_cnt = {}

    def process(self, line):
        tokens = line.strip().split()
        if len(tokens) == 1:
            self.doc_cnt += int(tokens[0])
        elif len(tokens) == 2:
            self.idf_cnt[tokens[0]] = self.idf_cnt.get(tokens[0],0) + int(tokens[1])
        else:
            print 'reduce input invalid, input:%s' % line
            sys.exit(0)


    def done(self):
        count = 0
        self.output.write('%s\n' % self.doc_cnt)
        for token in self.idf_cnt:
            if self.idf_cnt[token] > self.params.get("min_count",0):
                self.output.write('%s %s %s\n' %(count, token, self.idf_cnt[token]))
                count += 1
        self.output.close()
