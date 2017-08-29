#coding=utf8
import sys, os, multiprocessing
import time
import mr.mr as mr
from util.parser import Parser

class ParserMapper(mr.Mapper):
    def __init__(self, rank, file_list, output_dir, params):
        super(ParserMapper, self).__init__(rank, file_list, output_dir, params)
        self.parser = Parser()

    def process(self, line):
        tokens = line.strip().split()
        label = tokens[0]
        words = self.parser.parse(' '.join(tokens[1:]))
        self.output.write(label + ' ' + ' '.join(words) + '\n')

    def done(self):
        self.output.close()

