#coding=utf8
import sys, os, multiprocessing
import time
from util import util

class Mapper(multiprocessing.Process):
    def __init__(self, rank, file_list, output_dir, params):
        super(Mapper, self).__init__()
        self.rank = rank
        self.file_list = file_list
        self.output = open(output_dir + '/' + str(self.rank), 'w')
        self.params = params
        if 0 == rank:
            print 'map params:', self.params

    def process(self, line):
        self.output.write(line)

    def done(self):
        self.output.close()
        
    def run(self):
        count = 0
        for file_ in self.file_list:
            count += 1
            print 'mapper:%s processing %s, %s/%s' % (self.rank, file_, count, len(self.file_list))
            for line in open(file_):
                self.process(line)
        self.done()


class Reducer(object):
    def __init__(self, file_list, output_dir, params, output_file=None):
        self.file_list = file_list
        util.mkdir(output_dir)
        self.params = params
        if output_file:
            self.output = open(output_dir + '/' + output_file,'w')
        else:
            self.output = open(output_dir + '/' + str(int(time.time())),'w')
        print 'reduce params:', self.params

    def process(self, line):
        self.output.write(line)

    def done(self):
        self.output.close()

    def run(self):
        count = 0
        for file_ in self.file_list:
            count += 1
            print 'reducer processing %s, %s/%s' % (file_, count, len(self.file_list))
            for line in open(file_):
                self.process(line)
        self.done()
