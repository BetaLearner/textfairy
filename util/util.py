import sys, os
import shutil

def mkdir(dir_path, delete_if_exist=True):
    if delete_if_exist and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def get_inst(module_class):
    tokens = module_class.strip().split('.')
    return getattr(sys.modules['.'.join(tokens[:-1])], tokens[-1])

    
    
