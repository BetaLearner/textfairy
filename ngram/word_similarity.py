#encoding=utf8
import sys, heapq, random
from threading import Thread
from scipy import spatial
import numpy as np

word2vec = {}
topk_similarity_words = []
def load_word2vec(infile):
    global word2vec
    for line in open(infile):
        word,vec = line.strip().split('\t')
        zero_count = 0
        for v in vec.split(','):
            if float(v) == 0.0:
                zero_count += 1
        if zero_count >= len(vec.split(','))/2:
            continue
        word2vec[word] = np.array([float(v) for v in vec.split(',')])
    print 'load word2vec success, len: ', len(word2vec)


def topk_similarity(word, topk):
    heap_data = []
    max_score = -1.0
    for w in word2vec:
        if w == word:
            continue
        score = 1 - spatial.distance.cosine(word2vec[word],word2vec[w])
        if len(heap_data) < topk:
            heapq.heappush(heap_data, (score,w))
        else:
            if score > heap_data[0][0]:
                heapq.heapreplace(heap_data,(score,w))
    for item in heapq.nlargest(topk,heap_data):
        print item[0],item[1]

def topk_similarity_thread(word, topk, start_idx, end_idx, thread_id):
    heap_data = []
    max_score = -1.0
    for w in word2vec.keys()[start_idx:end_idx]:
        if w == word:
            continue
        score = 1 - spatial.distance.cosine(word2vec[word],word2vec[w])
        if len(heap_data) < topk:
            heapq.heappush(heap_data, (score,w))
        else:
            if score > heap_data[0][0]:
                heapq.heapreplace(heap_data,(score,w))
    global topk_similarity_words
    topk_similarity_words[thread_id] = heap_data

def merge_result(topk):
    global topk_similarity_words
    heap_data = []
    for sw in topk_similarity_words:
        for item in sw:
            heapq.heappush(heap_data, (item[0],item[1]))
    for item in heapq.nlargest(topk,heap_data):
        print item[0],item[1]
             
def word_similarity(word1,word2):
    global word2vec
    if word1 not in word2vec or word2 not in word2vec:
        print 'not found'
    else:
        print word1,word2,1 - spatial.distance.cosine(word2vec[word1],word2vec[word2])          
load_word2vec(sys.argv[1])
thread_num = 6
while True:
    '''
    words = sys.stdin.readline().strip().split(' ')
    if len(words) != 2:
        print 'input error.'
        continue
    print words
    word_similarity(words[0],words[1])
    continue
    '''
    topk_similarity_words = []
    word = sys.stdin.readline().strip()
    if word not in word2vec:
        print word + ' not in word2vec model.'
        continue
    if len(word) == 0:
        continue
    try:
        step = (len(word2vec)+thread_num) / thread_num
        print step
        threads = []
        for i in range(thread_num):
            thread = Thread( target=topk_similarity_thread, args=(word,10,step*i,step*i+step,i) )
            threads.append(thread)
            topk_similarity_words.append([])
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        merge_result(10)   
        #topk_similarity(word,10)
    except Exception as e:
        print e


