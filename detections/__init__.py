# Detectionクラス
# -------------------------------------------------------------

import numpy as np
from utility.imgproc import tlbr2tlwh, tlwh2tlbr

from vendors.deep_sort import deep_sort
from deep_sort.detection import Detection as Original

class Detection(Original):
    """
    画像から検出した各種コンテキストを管理するクラス

    Arguments:
        box: 人物検出結果（det.txtの中身1行分）
        tsn: 人物検出結果に対応するTSN出力結果（XX_txn.txtの中身1行分）
        cnn: 人物検出結果に対応するCNN特徴（cnn.txtの中身1行分）
        **kwargs: その他、パラメータ(必要なもの)
    """

    def __init__(self, box, tsn, cnn, **kwargs):
        # スーパークラス初期化
        tlwh = tlbr2tlwh(box['box'])

        confidence = float(box['score'])
        features = cnn['features']
        super().__init__(tlwh, confidence, features)

        # その他のデータセット
        self.tsn = tsn

    def __dict__(self):
        return {
            'box': self.tlwh,
            'score': self.confidence,
            'features': self.features,
            'tsn': self.tsn
        }

    @property
    def tsn_score(self):
        return self.tsn['scores']
