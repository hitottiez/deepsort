# 各種画像関連処理
# ------------------------------------------------

import numpy as np
import cv2


def tlbr2tlwh(box):
    """
    [left, top, right, bottom]から[left, top, width, height]に変換する
    ::
    """
    return np.asarray([
        box[0],
        box[1],
        box[2]-box[0],
        box[3]-box[1]
    ], dtype=np.float)


def tlwh2tlbr(box):
    """
    [left, top, width, height]から[left, top, right, bottom]に変換する
    """
    return np.asarray([
        box[0],
        box[1],
        box[0] + box[2],
        box[3] + box[1]
    ], dtype=np.float)


def softmax(raw_score, T=1):
    """
    ソフトマックス関数
    """
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None]) * T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]
