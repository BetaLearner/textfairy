#coding=utf-8
import multiprocessing
import sys, time

class FileLoader(multiprocessing.Process):
    def __init__(self, file_path, queue, scan_num=1):
        super(FileLoader, self).__init__()
        self.file_path = file_path
        self.queue = queue
        self.scan_num = scan_num
        
    def run(self):
        num = 0
        while num < self.scan_num:
            for line in open(self.file_path):
                self.queue.put(line)
            num += 1

class DataCache():
    def __init__(self, file_path, scan_num=1, cache_size=500000):
        self.queue = multiprocessing.Queue(cache_size)
        self.loader_process = FileLoader(file_path, self.queue, scan_num)
        self.loader_process.daemon = True
        self.loader_process.start()

    def has_next(self):
        return self.queue.size() > 0 or self.loader_process.is_alive()

    def next_batch(self, bs=100):
        labels, instances = [], []
        retry_num = 0
        while len(labels) < bs and retry_num < 3:
            try:
                line = self.queue.get_nowait()
                tokens = line.strip().split()
                label = int(tokens[0])
                if label == -1:
                    label = 0
                inst = {}
                for token in tokens[1:]:
                    k,v = token.split(':')
                    inst[k] = float(v)
                if inst:
                    labels.append(label)
                    instances.append(inst)
            except Exception as e:
                if not self.loader_process.is_alive() and self.queue.empty():
                    retry_num += 1
        return labels, instances


if __name__ == '__main__':
    data = DataCache(sys.argv[1])
    #print len(data.next_batch()[0])
   
