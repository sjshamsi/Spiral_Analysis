import pickle
import os

def save_dict(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def append_files(dir_path, ext='', ret_path=False):
    file_list = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(ext):
                if ret_path:
                    file_list.append(root + file)
                else:
                    file_list.append(file)
    return file_list