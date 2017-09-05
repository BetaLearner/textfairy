import sys
import jieba
import jieba.posseg as pseg

while True:
    sentence = sys.stdin.readline().strip()
    print sentence
    #words = jieba.cut(sentence, cut_all=True)
    words = jieba.cut_for_search(sentence)
    words = [word for word in words if word != ' ']
    print ' '.join(words)
