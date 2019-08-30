# Okutamaデータセットの人物追跡結果評価PG
# ------------------------------------------------------------------------

import argparse
import os
import pickle

import numpy as np

import __addpath
import motmetrics as mm
import pandas as pd
from tools.eval_utils.context_dataset import ContextDataSet
from tools.eval_utils.load_vatic_label import load_gt
from vendors.deep_sort import deep_sort
from deep_sort.iou_matching import iou
from natsort import natsorted

def _tlbr_to_tlwh(bbox):
    bbox[2] -= bbox[0]
    bbox[3] -= bbox[1]
    return bbox

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_root', help='ground truth directory. ex) /path/to/dataset/labelst/test')
    parser.add_argument('--save_root', help='データセット識別結果ディレクトリ')
    parser.add_argument('--iou_cost_th', type=float,
                        default=0.5, help='指定したコスト以上のデータはマッチしなかったとみなす')

    # 個別指定によって集計したい項目をここにセットしてください
    # https://github.com/cheind/py-motmetrics
    # metrics = [
    #     'idf1',
    #     'idp',
    #     'idr',
    #     'recall',
    #     'precision'
    # ]
    metrics = mm.metrics.motchallenge_metrics

    args = parser.parse_args()

    logs = ContextDataSet(args.save_root)
    mh = mm.metrics.create()

    # 出力
    df = None

    for video_name, context in logs.context_logs.items():
        print('Summary: {}'.format(video_name))
        gt = load_gt(args.gt_root, video_name, dataset_type='okutama')

        # ['1.1.9', '2.2.1', '1.1.8'] はフレーム数とgtの数が一致しない
        # （人がいなくなってから動画の最後までのフレームのgtがない
        # ので、スキップさせずに続ける。
        if len(gt) != len(context) and video_name not in ['1.1.9', '2.2.1', '1.1.8']:
            print('len(gt) {} != len(context) {} のためスキップ'.format(
                len(gt), len(context)))
            continue
        acc = mm.MOTAccumulator(auto_id=True)
        for frame_idx in range(len(gt)):
            # 正解と検出結果を取り出す
            gt_ids, gt_boxes = zip(*[(g['local_id'], _tlbr_to_tlwh(g['box']))
                                     for g in gt[frame_idx]]) if len(gt[frame_idx]) > 0 else([], [])
            det_ids, det_boxes = zip(*[(d['local_id'], _tlbr_to_tlwh(d['location']))
                                       for d in context[frame_idx]['context']]) if len(context[frame_idx]['context']) > 0 else ([], [])
            candidates = np.asarray(det_boxes)
            gt_boxes = np.asarray(gt_boxes)
            costs = []
            for gt_box in gt_boxes:
                if len(candidates) == 0:
                    costs.append([])
                else:
                    cost = 1. - iou(gt_box, candidates)
                    cost = list(map(lambda x: np.nan if x >=
                                    args.iou_cost_th else x, cost))
                    costs.append(cost)
            acc.update(gt_ids, det_ids, costs)
        # 集計
        df_sub = mh.compute(acc, metrics=metrics, name=video_name)
        if df is None:
            df = df_sub
        else:
            df = pd.concat([df, df_sub])
        # motmetric出力
        pkl_filepath = os.path.join(args.save_root, video_name, 'motacc.pkl')
        with open(pkl_filepath, 'wb') as f:
            pickle.dump(acc, f)

    # 出力
    df = df.reindex(index=natsorted(df.index))
    print(df)
    filepath = os.path.join(args.save_root, 'motmetric.csv')
    with open(filepath, 'w') as f:
        f.write(df.to_csv())
