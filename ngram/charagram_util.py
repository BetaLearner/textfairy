import os

def createDir(path):
    if os.path.isfile(path):
        print path, ' is exist file'
        sys.exit(0)
    if not os.path.isdir(path):
        os.mkdir(path)

def filt(line):
    infos = line.split('\t')
    if len(infos) != 3:
        return False
    if len(infos[1].split(' ')) < 2 or len(infos[2].split(' ')) < 5:
        return False
    return True

def getFileLineNum(filepath, use_filt=False):
    count = 0
    try:
        for line in open(filepath,'r'):
            if use_filt and filt(line) == False:
                continue
            count += 1
    except Exception as e:
        print e
    return count

def load_dict(dict_path, threshold=1):
    vocab_map = {}
    for line in open(dict_path):
        kv = line.strip().split('\t')
        if len(kv) != 3:
            print 'error', line
            continue
        if int(kv[2]) >= threshold:
            vocab_map[kv[0]] = int(kv[1])
    vocab_map['patting'] = len(vocab_map) + 1
    print dict_path, 'dict len: ', len(vocab_map)
    return vocab_map
