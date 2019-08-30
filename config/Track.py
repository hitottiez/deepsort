# -*- coding: utf-8 -*-

# API設定管理クラス
# ==============================================================================

from .ISection import ISection
from .conf_util import createpath, get_project_root, boolcast
import numpy as np
import logging
logger = logging.getLogger(__file__)


class Track(ISection):
    """
    Track関連の設定を管理するクラス
    """

    def __init__(self, ini):
        super().__init__(ini)

    @property
    def cnn_metric_type(self):
        """
        CNN特徴量マッチングの指標
        """
        return self.ini.get('track', 'cnn_metric_type')

    @property
    def cnn_matching_th(self):
        """
        CNN特徴量マッチングしきい値
        """
        return self.ini.getfloat('track', 'cnn_matching_th')

    @property
    def max_iou_distance(self):
        """
        Trackerパラメータ (IoU距離しきい値)
        """
        return self.ini.getfloat('track', 'max_iou_distance')

    @property
    def max_age(self):
        """
        Trackerパラメータ (Trackの最大経過フレーム数)
        """
        return self.ini.getint('track', 'max_age')

    @property
    def n_init(self):
        """
        Trackerパラメータ (動線を有効にする際のしきい値)
        """
        return self.ini.getint('track', 'n_init')

    @property
    def enable_tsn_matching(self):
        """
        行動によるマッチング有効／無効
        """
        return boolcast(self.ini.get('track', 'enable_tsn_matching'))

    @property
    def matching_action_topk(self):
        """行動マッチングでチェック対象とする上位N件のNを返却"""
        return self.ini.getint('track', 'matching_action_topk')

    @property
    def max_action_distance(self):
        """行動マッチングで使用するコストの最大距離"""
        return self.ini.getfloat('track', 'max_action_distance')

    @property
    def tsn_segment_size(self):
        """
        TSNセグメント
        """
        return self.ini.getint('track', 'tsn_segment_size')

    @property
    def tsn_score_th(self):
        """
        TSNスコアしきい値
        """
        return self.ini.getfloat('track', 'tsn_score_th')
