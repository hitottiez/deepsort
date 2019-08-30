# 複数人物の行動認識エンジン用Trackerクラス
# ※オリジナルのTrackerクラスを継承して作成
# -----------------------------------------------------------------

from vendors.deep_sort import deep_sort
from deep_sort.tracker import Tracker as Original
from deep_sort import nn_matching
from deep_sort.linear_assignment import matching_cascade

from deepsort.track import Track
from deepsort.action_matcher import ActionMatcher
import numpy as np
import logging
logger = logging.getLogger(__file__)


class Tracker(Original):
    cnn_metric_type = 'cosine'          # CNN特徴量でのマッチングパラメータ
    cnn_matching_th = 0.2               # CNN特徴量マッピングで使うしきい値
    max_iou_distance = 0.7              # IoUマッチングで使うしきい値
    max_age = 30                        # Trackのmax_age
    n_init = 3                          # Trackのn_init
    matching_action_topk = 5            # 行動によるマッチング処理で使用する、評価する行動数の上位n件
    matching_action_mat = None          # 行動マッチング処理で使用する(予定)の行列データ
    max_action_distance = 0.7           # 行動によるマッチング処理で使用する、距離しきい値
    max_person_distance = 0.7           # 行動によるマッチング処理で使用する、距離しきい値
    enable_tsn_matching = True          # Falseにしたら行動によるマッチング処理はスキップ

    def __init__(self):
        cnn_metric = nn_matching.NearestNeighborDistanceMetric(self.cnn_metric_type, self.cnn_matching_th)
        super().__init__(cnn_metric, self.max_iou_distance, self.max_age, self.n_init)
        self.action_matcher = ActionMatcher(self.matching_action_topk, self.matching_action_mat)

    def predict(self):
        super().predict()

    def update(self, detections):
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        
        # 既存のTrackを更新
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        
        # マッチしなかったtrackを更新
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # マッチしなかったDetectionsでTrackを新規作成
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # 未検出が続いたTrackを削除
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # カルマンフィルタ更新.
        self._update_distance_metric()

    def _match(self, detections):
        #import pdb; pdb.set_trace()
        matches, unmatched_tracks, unmatched_detections = super()._match(detections)
        
        matches, unmatched_tracks, unmatched_detections = \
            self._match_tsn(detections, matches, unmatched_tracks, unmatched_detections)
        
        return matches, unmatched_tracks, unmatched_detections

    def _match_tsn(self, detections, matches, unmatched_tracks, unmatched_detections):
        """
        行動によるマッチング処理
        """
        if not self.enable_tsn_matching:
            return matches, unmatched_tracks, unmatched_detections
        
        matches_a, unmatched_tracks_a, unmatched_detections_a = \
            matching_cascade(self.action_matcher,
                             self.max_action_distance,
                             self.max_age,
                             self.tracks,
                             detections,
                             unmatched_tracks,
                             unmatched_detections)
        matches = matches + matches_a
        if len(matches_a) > 0:
            logger.debug('行動認識結果でマッチングされました')

        return matches, unmatched_tracks_a, unmatched_detections_a

    def _initiate_track(self, detection):
        """
        新規Trackを作成する処理
        """
        mean, covariance = self.kf.initiate(detection.to_xyah())
        track = Track(mean, covariance,
                      self._next_id, self.n_init, self.max_age,
                      detection.feature,
                      detection.tsn_score)
        self.tracks.append(track)
        self._next_id += 1

    def _update_distance_metric(self):
        """CNN特徴更新"""
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

