#encoding=utf8
import sys, os, random
#sys.path.insert(0, '/opt/tiger/push/python_package/lib/python2.7/site-packages')
from keras.layers import Activation, Flatten, Dropout, Input, Embedding, SimpleRNN, GRU, LSTM, Dense, merge, Bidirectional
from keras.models import Model
from keras.layers.core import Lambda
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.engine.topology import Layer
from keras import backend as K
import numpy as np
import charagram_util as cutil
import charagram_model as cm
import sampler, keras

dict_threshold = 2
patting = 'patting'

class MeanLayer(Layer):
    def __init__(self, mask_zero=False, **kwargs):
        super(MeanLayer, self).__init__(**kwargs)
        self.mask_zero = mask_zero

    def build(self, input_shape):
        self.trainable_weights = []

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
    
    def call(self, x, mask=None):
        return K.mean(x, axis=1) # look out

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1]) # look out

def setRow(mat, row, vals, maxlen):
    count = 0
    for val in vals:
        if count < maxlen:
            mat[row][count] = val
        else:
            break
        count += 1
    
def load_title_content(news_data_file, vocab_map, maxlen = 100, test_ratio = 0.3):
    total_news = cutil.getFileLineNum(news_data_file, use_filt=True)
    train_num = int(total_news * (1-test_ratio))
    train_t = np.zeros((train_num,maxlen)) 
    train_c = np.zeros((train_num,maxlen))
    test_t = np.zeros((total_news-train_num, maxlen))
    test_c = np.zeros((total_news-train_num, maxlen))
    count = 0
    filt_count = 0
    for line in open(news_data_file):
        if cutil.filt(line) == False:
            filt_count += 1
            continue
        doc = line.strip().split('\t')
        title = [vocab_map[word] if word in vocab_map else vocab_map[patting] for word in doc[1].split(' ')]
        content = [vocab_map[word] if word in vocab_map else vocab_map[patting] for word in doc[2].split(' ')]
        if count < train_num:
            setRow(train_t, count, title, maxlen)
            setRow(train_c, count, content, maxlen)
        elif count < total_news:
            setRow(test_t, count-train_num, title, maxlen)
            setRow(test_c, count-train_num, content, maxlen)
        count += 1
    print 'filt_count:',filt_count
    return train_t, train_c, test_t, test_c
        
def model_build(vocab_size, init_embedding_file, dict_path, layerType, maxlen):
    embedding_size = 64
    hidden_layer_size = 128
    title = Input(shape=(maxlen,), dtype='int32')
    content = Input(shape=(maxlen,), dtype='int32')
    if init_embedding_file:
        shared_embedding = Embedding(vocab_size+2, embedding_size, input_length=maxlen, init='glorot_normal', mask_zero=True,
                weights=[cm.load_embedding_file(init_embedding_file,dict_path, dict_threshold)])
    else:
        shared_embedding = Embedding(vocab_size+2, embedding_size, input_length=maxlen, init='glorot_normal', mask_zero=True)
    #shared_embedding = Embedding(vocab_size+2, embedding_size, input_length=maxlen, init='glorot_normal', mask_zero=True)
    map_title = shared_embedding(title)
    map_content = shared_embedding(content)
    if layerType == 'mean':
        encoded_title = MeanLayer(True)(map_title)
        encoded_content = MeanLayer(True)(map_content)
    elif layerType == 'bilstm':
        encoded_title = Bidirectional(LSTM(hidden_layer_size))(map_title)
        encoded_content = Bidirectional(LSTM(hidden_layer_size))(map_content)
    else:
        encoded_title = LSTM(hidden_layer_size)(map_title)
        encoded_content = LSTM(hidden_layer_size)(map_content)
    merged_vector = merge([encoded_title, encoded_content], mode='dot')# dot_axes=1) ## dot_axes ??
    predictions = Dense(1, activation='sigmoid')(merged_vector)
    model = Model(input=[title, content], output=predictions)
    adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08)
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model, shared_embedding

def model_train(news_data_file, dict_path, init_embedding_file, model_save_path, pos_size=50, epoch=10, maxlen=50, layerType='bilstm'):
    #load data
    vocab_map = cutil.load_dict(dict_path, dict_threshold)
    train_t, train_c, test_t, test_c = load_title_content(news_data_file, vocab_map, maxlen=maxlen)
    print 'load data success.'
    #build model
    model, shared_embedding =  model_build(len(vocab_map), init_embedding_file, dict_path, layerType, maxlen)
    #train model
    for i in range(epoch):
        #shuffle data
        bias = 0
        while bias < len(train_t):
            sample_t, sample_c, labels = sampler.negativeSampling(train_t, train_c, bias, pos_size)
            model.fit([sample_t, sample_c], labels, batch_size=50, nb_epoch=1)
            bias += pos_size
        sample_t, sample_c, labels = sampler.negativeSampling(test_t, test_c, 0, len(test_t))
        score = model.evaluate([sample_t, sample_c], labels, batch_size=1000)
        print 'iter: ', i, 'test: ', score
        model.save_weights('charagram.model')

    model.load_weights('charagram.model')    
    sample_t, sample_c, labels = sampler.negativeSampling(train_t, train_c, 0, len(train_t))
    score = model.evaluate([sample_t, sample_c], labels, batch_size=1000)
    print 'training: ', score
    sample_t, sample_c, labels = sampler.negativeSampling(test_t, test_c, 0, len(test_t))
    score = model.evaluate([sample_t, sample_c], labels, batch_size=1000)
    print 'test: ', score
    cm.save_embedding_file(shared_embedding.get_weights(), dict_path, model_save_path)

def get_predict_data(title, content, row_size, col_size):
    title_mat = np.zeros((row_size,col_size))
    content_mat = np.zeros((row_size,col_size))
    for row in range(row_size):
        col = 0
        for tid in title[:col_size]:
            title_mat[row][col] = tid
            col += 1
        content_mat[row][0] = content[row]
        
    return title_mat, content_mat

def get_top_keywords(news_data_file, dict_path, model_path, topk=10, layerType='bilstm', maxlen=50):
    vocab_map = cutil.load_dict(dict_path,dict_threshold)
    vocab_reverse = {v:k for k,v in vocab_map.items()}
    #train_t, train_c, test_t, test_c = load_title_content(news_data_file, vocab_map, maxlen=maxlen)
    model, shared_embedding =  model_build(len(vocab_map), None, dict_path, layerType, maxlen)
    model.load_weights(model_path)
    for line in open(news_data_file):
        if cutil.filt(line) == False:
            continue
        doc = line.strip().split('\t')
        title = [vocab_map[word] if word in vocab_map else vocab_map[patting] for word in doc[1].split(' ')]
        content = [vocab_map[word] if word in vocab_map else vocab_map[patting] for word in doc[2].split(' ')]
        title_r, content_r = get_predict_data(title, content, min(len(content),maxlen), maxlen)
        scores = model.predict([title_r, content_r])
        wordid2score = {content[idx]:scores[idx][0] for idx in range(len(scores))}
        wordid2score_sort = sorted(wordid2score.items(), key=lambda x:x[1], reverse=True)
        out_str = []
        for item in wordid2score_sort[:topk]:
            out_str.append(str(vocab_reverse[item[0]])+':'+str(item[1]))   
        print doc[0],'\t','\t'.join(out_str)

if __name__ == '__main__':
    if not (len(sys.argv) == 6 or len(sys.argv) == 4):
        print 'Usage:\n\tpython %s news_data_file, dict_path, init_embedding_file, embedding_save_path, layerType'
        print '\nor\n\tpython %s news_data_file, dict_path, model_path'
        sys.exit()
    model_train(sys.argv[1],sys.argv[2], sys.argv[3], sys.argv[4], pos_size=1000, epoch=10, maxlen=50, layerType=sys.argv[5])
    #get_top_keywords(sys.argv[1], sys.argv[2], sys.argv[3])
