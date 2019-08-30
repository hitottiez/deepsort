# VATIC形式のラベルをロードする

import os

import pandas as pd
from app.action_master import ActionMaster


def load_gt(gt_root, videoname, dataset_type='okutama'):
    # -2はいらない
    master = ActionMaster(dataset_type)
    del master.__dict__()[-2]

    labelMap = master.getNameMap()
    no_action = master.findByModelID(-1)
    labelfilepath = os.path.join(gt_root, '{}.txt'.format(videoname))
    df = pd.read_csv(labelfilepath, sep=' ',
                     names=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    num_frames = df[5].max()
    df = df[(df[6] != 1) & (df[7] != 1)]
    df[10] = df[10].fillna(no_action)
    gt = [[] for x in range(num_frames)]
    for frame_idx, frame_df in df.groupby(5):
        if frame_idx == 0:  # frame_idx=0は検出結果がないのでスキップ(オプティカルフロー的な問題で)
            continue
        frame_idx = frame_idx - 1
        for local_id, local_id_df in frame_df.groupby(0):
            box = [int(x) for x in local_id_df.iloc[0, 1:5].tolist()]
            classes = local_id_df.loc[:, 10:13].values.tolist()[0]
            classes = [(labelMap[x], x) for x in classes if x in labelMap]
            gt[frame_idx].append({
                'local_id': local_id,
                'box': box,
                'action_id': [x[0] for x in classes],
                'action_name': [x[1] for x in classes],
            })
    return gt
