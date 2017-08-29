#coding=utf8
import sys, os, multiprocessing
import logging, logging.handlers, json
from util import util

class Scheduler():
    def __init__(self, conf_file):
        self.conf = json.loads(open(conf_file).read())
        logger = logging.getLogger(self.conf.get('job_name'))
        log_handler = logging.handlers.RotatingFileHandler(self.conf['log_file'], maxBytes = 1024*1024)
        logger.addHandler(log_handler)

    def schedule(self):
        logging.info('push predict dispatch main process start')

        for job in self.conf.get("jobs"):
            print 'process job:%s' % job['job_name']
            __import__(job['map_module'])
            mapper_inst = getattr(sys.modules[job['map_module']], job['mapper'])
            reducer_inst = None
            if job.get('reducer','NONE') != 'NONE':
                __import__(job['reduce_module'])
                reducer_inst = getattr(sys.modules[job['reduce_module']], job['reducer'])
            #map
            input_files = [job['input_dir'] + '/' + file_ for file_ in os.listdir(job['input_dir'])]
            print ','.join(input_files)
            sys.stdout.flush()
            ps = []
            if job['map_num'] > len(input_files):
                job['map_num'] = len(input_files)
            file_num = len(input_files)/job['map_num'] if len(input_files) % job['map_num'] == 0 else len(input_files)/job['map_num'] + 1
            util.mkdir(job['map_output_dir'],delete_if_exist=True)
            for i in range(job['map_num']):
                p = mapper_inst(i, input_files[i*file_num:(i+1)*file_num], job['map_output_dir'], job.get("map_params",{}))
                p.daemon = True
                p.start()
                ps.append(p)
            for p in ps:
                p.join()
            print 'map output_dir:', job['map_output_dir']
            #reduce
            output_file = job.get('output_file',None)
            if reducer_inst:
                input_files = [job['map_output_dir'] + '/' + file_ for file_ in os.listdir(job['map_output_dir'])]
                reducer = reducer_inst(input_files, job['output_dir'], job.get("reduce_params",{}), output_file)
                reducer.run()
                print 'reduce output_dir: ', job['output_dir']

        
if __name__ == '__main__':
    scheduler = Scheduler(sys.argv[1])
    scheduler.schedule()
