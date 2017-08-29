#coding=utf-8
import jieba

class Parser():
    def __init__(self, dict_path=''):
        return 

    def cut(self, sentence):
        return jieba.cut(sentence, cut_all=False)


if __name__ == '__main__':
    parser = Parser()
    for word in parser.cut('我来到北京清华大学'):
        print word
