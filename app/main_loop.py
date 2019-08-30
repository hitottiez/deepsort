# メインループ処理

import time
import datetime
import json
import os
import warnings

import cv2

from app import config
from deepsort import Track
from deepsort.tracker import Tracker
from detections import Detection


# Trackのパラメータセット
Track.tsn_segment = config.track.tsn_segment_size
Track.tsn_score_threshold = config.track.tsn_score_th
# Trackerのパラメータセット
Tracker.cnn_metric_type = config.track.cnn_metric_type
Tracker.cnn_matching_th = config.track.cnn_matching_th
Tracker.max_iou_distance = config.track.max_iou_distance
Tracker.max_age = config.track.max_age
Tracker.n_init = config.track.n_init
Tracker.matching_action_topk = config.track.matching_action_topk
Tracker.max_action_distance = config.track.max_action_distance
Tracker.enable_tsn_matching = config.track.enable_tsn_matching

# ignore sklearn deprecation warning
warnings.filterwarnings("default", category=DeprecationWarning)


def get_timestamp():
    """
    現在時刻のタイムスタプを返す
    """
    now = datetime.datetime.now()
    timestamp = int(time.mktime(now.timetuple()) * 1e3 + now.microsecond / 1e3)
    return timestamp


class MainLoop:
    def __init__(self, writer, master, image_filepath):
        """
        メインループ処理

        Arguments:
            writer: ライタークラス
            master: 行動マスタ
            image_filepath: 動画のjpgファイルが含まれるディレクトリパス
        """
        # 設定とか
        self.writer = writer
        self.master = master
        self.image_filepath = image_filepath

        # cnn.txt tsn.txt, det.txt
        self.features_base_path = os.path.join(image_filepath, 'feature_results')

        # Tracker作成
        self.tracker = Tracker()

    def __call__(self, tsn_modality=None):

        print('Start MainLoop.')

        tsn_f = open(os.path.join(self.features_base_path, '{}_tsn.txt'.format(tsn_modality)))
        cnn_f = open(os.path.join(self.features_base_path, 'cnn.txt'))
        with open(os.path.join(self.features_base_path, 'det.txt'), 'r') as f:
            lines = f.readlines()

        print('processing {}'.format(self.image_filepath))
        frame_idx = -1
        for line in lines:

            tsn = tsn_f.readline()
            cnn = cnn_f.readline()
            frame_idx += 1

            if frame_idx == 0:  # 1フレーム目はスキップ
                continue

            timestamp = get_timestamp()
            _, boxes = line.split(' ')  # no use image_filename
            boxes = json.loads(boxes)
            tsn = json.loads(tsn)
            cnn = json.loads(cnn)

            detections = [Detection(box, tsn[idx], cnn[idx])
                          for idx, box in enumerate(boxes)]

            # トラッカー更新
            tracker = self._update_track(detections=detections)

            # データ出力
            # DeepSORTオリジナルに合わせて、確定動線のうち、time_sice_updateが1までの追跡を出力対象とする
            # https://github.com/nwojke/deep_sort/blob/280b8bdb255f223813ff4a8679f3e1321b08cdfc/deep_sort_app.py#L194
            outputs = [t.output() for t in tracker.tracks if t.is_confirmed()
                       and t.time_since_update <= 1]

            for context in outputs:  # 行動名を付与
                act_labels = [self.master.findByModelID(
                    model_id) for model_id in context['action']['model_id']]
                context['action']['label'] = act_labels

            # 出力処理(コンテキストデータ)
            self.writer.writeContextData(frame_idx, timestamp, outputs)

            # 出力処理(動画)
            filepath = os.path.join(
                self.image_filepath, '{:d}.jpg'.format(frame_idx))
            image = cv2.imread(filepath)
            self.writer.writeFrame(image, outputs)

        print('Finish MainLoop.')

    def _update_track(self, detections):
        self.tracker.predict()
        self.tracker.update(detections)
        return self.tracker
