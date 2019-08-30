# 行動によるマッチングコスト計算
# ------------------------------------------------------

import numpy as np
from vendors.deep_sort import deep_sort
from deep_sort.linear_assignment import INFTY_COST
from utility.imgproc import softmax

class ActionMatcher:

    def __init__(self, top_k=5, action_mat=None):
        self.top_k = top_k
        self.action_mat = action_mat

    def __call__(self, tracks, detections, track_indices=None, detection_indices=None):
        """
        行動によるコスト計算
        """
        if track_indices is None:
            track_indices = np.arange(len(tracks))
        if detection_indices is None:
            detection_indices = np.arange(len(detections))

        # detection取得
        target_detections = [self.__extract_top5(detections[i]) for i in detection_indices]

        # コスト計算
        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            action_id, score = track.to_action(topk=1)
            if track.time_since_update > 1 or action_id < 0:
                cost_matrix[row, :] = INFTY_COST
                continue
            cost_matrix[row, :] = self.__calc_cost(action_id, target_detections)
        return cost_matrix

    def __calc_cost(self, action_id, target_detections):
        # 行動IDが上位5件に含まれていたら、1-スコアをコストとする
        # 含まれていないならコスト無限大とする
        a = [1. - t[action_id] if action_id in t else INFTY_COST for t in target_detections]
        return np.array(a)

    def __extract_top5(self, det):
        scores = np.array(det.tsn_score)
        scores = softmax(scores)
        top5 = np.argsort(scores)[::-1][:self.top_k]
        return dict([(idx, scores[idx]) for idx in top5])
