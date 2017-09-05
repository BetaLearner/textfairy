#-*- coding: UTF-8 -*-
import jieba.posseg as pseg
import jieba
import sys, multiprocessing, os
import charagram_util as cutil

g_punctations = set(['!',',','.','(',')','...','?','！','，','。','。。。','（','）','=','+','_','？'])
def remove_punctation_sent(line):
    word_pos_list = pseg.cut(line)
    ret = ''
    pre = ''
    for word, pos in word_pos_list:
        if pos == 'x':# or pos == 'm':# x: puntation, m: liangci
            if pre != ' ':
                ret += ' punction '
                pre = ' '
        elif pos == 'm':
            if pre != ' ':
                ret += ' measure '
                pre = ' '
        else:
            ret += word.encode('utf8')
            pre = 'h'
    return ret
        
def remove_punctation_file(infile, outfile):
    fout = open(outfile,'w')
    for line in open(infile,'r'):
        if len(line.strip().split('\t')) != 3:
            continue
        gid, title, content = line.strip().split('\t')
        fout.write(gid+'\t'+remove_punctation_sent(title)+'\t'+remove_punctation_sent(content)+'\n')
    fout.close()

def content_clear(line):
    clear_set = set(['网讯'])
    for word in clear_set:
        idx = line.find(word)
        if idx != -1 and idx < 21:
            line = line[idx+len(word):]
    lidx = line.find('（')
    ridx = line.find('）',lidx)
    clear_set2 = set(['记者','通讯'])
    for word in clear_set2:
        idx = line.find(word)
        if idx > lidx and idx < ridx:
            line = line[:lidx] + ' ' + line[ridx+1:]
    return line 

def word_segment(line, out_dict, mode=2, clear=False):
    line = remove_punctation_sent(line)
    if clear:
        line = content_clear(line)
    if mode == 1:
        words = jieba.cut(line, cut_all=True)
    elif mode == 2:
        words = jieba.cut(line, cut_all=False)
    else:
        words = jieba.cut_for_search(line)
    words = [word.encode('utf8') for word in words if word.encode('utf8') != ' ']
    for word in words:
        out_dict.setdefault(word,0)
        out_dict[word] += 1
    return ' '.join(words)

def word_segment_file(infile, outfile, out_dict_file, mode=2):# mode = 3 search engine
    fout = open(outfile,'w')
    out_dict = {}
    for line in open(infile,'r'):
        if len(line.strip().split('\t')) != 3:
            continue
        gid, title, content = line.strip().split('\t')
        fout.write(gid+'\t'+word_segment(title, out_dict)+'\t'+word_segment(content, out_dict)+'\n')
    fout.close()
    fdict = open(out_dict_file, 'w')
    for word in out_dict:
        fdict.write(word+'\t'+str(out_dict[word])+'\n')
    fdict.close()

def word_filt(word):
    filt_set = set(['小姐','女士','先生'])
    for filt_word in filt_set:
        if word.endswith(filt_word) and len(word) == 3*len(filt_word)/2:
            return False
    return True

def multiProcessEngine(infile, outfile, out_dict_file, processNum):
    in_dir_tmp = 'in_dir_tmp'
    out_dir_tmp = 'out_dir_tmp'
    out_dict_dir_tmp = 'out_dict_dir_tmp'
    cutil.createDir(in_dir_tmp)
    cutil.createDir(out_dir_tmp)
    cutil.createDir(out_dict_dir_tmp)
    total_line = cutil.getFileLineNum(infile)
    fouts = [open(in_dir_tmp+'/'+infile+'_'+str(i),'w') for i in range(processNum)]
    count = 0
    for line in open(infile,'r'):
        fouts[count%processNum].write(line)
        count += 1
    for fout in fouts:
        fout.close()
    ps = []
    for i in range(processNum):
        file_name = infile+'_'+str(i)
        p = multiprocessing.Process(target=word_segment_file, args=(in_dir_tmp+'/'+file_name, out_dir_tmp+'/'+file_name, \
                    out_dict_dir_tmp+'/'+file_name, ))
        ps.append(p)
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    fout_result = open(outfile,'w')
    for i in range(processNum):
        for line in open(out_dir_tmp+'/'+infile+'_'+str(i),'r'):
            fout_result.write(line)
    fout_result.close()
    dict_all = {}
    for i in range(processNum):
        for line in open(out_dict_dir_tmp+'/'+infile+'_'+str(i),'r'):
            word, cnt = line.strip().split('\t')
            dict_all.setdefault(word,0)
            dict_all[word] += int(cnt)
    fout_dict = open(out_dict_file,'w')
    word_id = 1
    dict_sorted = sorted(dict_all.items(), key=lambda x:x[1], reverse=True)
    for item in dict_sorted:
        fout_dict.write(item[0] + '\t' + str(word_id) + '\t' + str(item[1]) + '\n')
        word_id += 1
    fout_dict.close()
    
#remove_punctation_file(sys.argv[1],sys.argv[2])
#word_segment_file(sys.argv[1], sys.argv[2], sys.argv[3])
multiProcessEngine(sys.argv[1], sys.argv[2], sys.argv[3], 20)
