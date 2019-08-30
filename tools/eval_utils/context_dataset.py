import json
import numpy as np
import os
import glob
from multiprocessing.pool import Pool

def _load_data(filepath):
    frames = []
    dirname = os.path.dirname(filepath)
    video_name = os.path.basename(dirname)
    print('load File: {}'.format(video_name))
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            frames.append(data)
    return video_name, frames

class ContextDataSet:

    def __init__(self, dirpath, worker=8):
        self.dirpath = dirpath
        files = glob.glob(os.path.join(self.dirpath, '*/contextlog.dat'))
        with Pool(worker) as p:
            logs = p.map(_load_data, files)
        self.context_logs = dict(logs)
        # mp_args = []
        # for filepath in sorted(files):
        #     self._load_data(filepath)

