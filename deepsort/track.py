# 複数人物特定のTrackクラス
# ------------------------------------------------------------------

from vendors.deep_sort import deep_sort
from deep_sort.track import Track as Original
from deep_sort.track import TrackState

import numpy as np
from utility.imgproc import softmax

import logging
logger = logging.getLogger(__name__)

class Track(Original):
    """
    Trackオブジェクト

    Arguments:
        mean: カルマンフィルタパラメータ
        covariance: カルマンフィルタパラメータ
        track_id: 割り当てるtrack_id
        n_init: 動線を生成する際に必要となる最低データ数
        max_age: 動線が消えるまでの最大欠損数
        feature: CNN特徴量データ
        tsn_scores: TSNスコア
    """

    # TSNセグメント
    tsn_segment = 30
    # TSNスコアしきい値
    tsn_score_threshold = 0.4

    def __init__(self, mean, covariance, track_id, n_init, max_age, 
                       feature=None, tsn_scores=None):
        super().__init__(mean, covariance, track_id, n_init, max_age, feature=feature)
        # TSNパラメータ
        self.__tsn_results = []
        if tsn_scores is not None:
            self.__tsn_results.append(tsn_scores)
   
    @property
    def tsn_results(self):
        return self.__tsn_results

    def calc_action_scores(self, alg='softmax'):
        """
        各行動のスコアを返却
        """
        tsn_raw_score = np.asarray(self.__tsn_results)
        tsn_raw_score = tsn_raw_score.mean(axis=0)

        if alg == 'softmax':
            return softmax(tsn_raw_score)

        # sigmoid
        return 1 / (1 + np.exp(-tsn_raw_score))

    def output(self):
        local_id = self.track_id
        location = self.to_tlbr()

        # okutama-action dataset, max multilabel num is 4
        model_ids, scores = self.to_action(topk=4)
        time_since_update = self.time_since_update
        action = {
            'model_id': model_ids,
            'score': scores
        }
        return {
            'local_id': local_id,
            'time_since_update': time_since_update,
            'hits': self.hits,
            'age': self.age,
            'location': location.tolist(),
            'action': action,
        }

    def to_action(self, topk=1):
        """
        行動認識結果を出力する

        Return:
            id: 行動認識結果ID(0始まり)
            score: 行動認識結果スコア
        """
        tsn_score_th = self.tsn_score_threshold

        # used deepsort cost
        if topk == 1:
            scores = self.calc_action_scores(alg='softmax')
            idx = np.argmax(scores)
            score = scores[idx]
            if len(self.__tsn_results) < self.tsn_segment or score < tsn_score_th:
                idx = -1
            return int(idx), score

        # used action mAP eval
        scores = self.calc_action_scores(alg='sigmoid')
        # descending sort and get top 4
        top4_idxes = np.argsort(-scores)[:topk]
        top4_scores = scores[top4_idxes]

        for i, score in enumerate(top4_scores):
            if score < self.tsn_score_threshold:
                top4_idxes[i] = -1

        return top4_idxes, top4_scores

    def predict(self, kf):
        """カルマンフィルタ更新"""
        super().predict(kf)

    def update(self, kf, detection):
        super().update(kf, detection)

        # 行動認識結果更新
        self.__update_action(detection)

    def __update_action(self, detection):
        """TSN結果更新"""
        segment = self.tsn_segment
        d_tsn = detection.tsn_score
        self.__tsn_results.append(d_tsn)
        self.__tsn_results = self.__tsn_results[-segment:]
