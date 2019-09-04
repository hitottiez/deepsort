#-*- coding: utf-8 -*-

# API共通関数
# ==============================================================================
import numpy as np
import tensorflow as tf

def tfsetup():
    """
    TensorFlowの初期化
    """
    tfconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=tfconfig)
    graph = tf.get_default_graph()
    return session, graph

def validate_box(image, box):
    """
    バウンディングボックスをチェックし、有効ならそのまま出力、無効ならNone出力
    """
    orgbox = box
    height, width = image.shape[:2]
    if type(box) != list or len(box) < 4:
        # boxフォーマットが違う
        logger.warn('Incorrect box format')
        return None
    # 順番補正
    box = [
        min(orgbox[0], orgbox[2]),
        min(orgbox[1], orgbox[3]),
        max(orgbox[0], orgbox[2]),
        max(orgbox[1], orgbox[3]),
    ]
    # 値の補正＋int型にする
    box[0] = int(box[0]) if box[0] >= 0 else 0
    box[1] = int(box[1]) if box[1] >= 0 else 0
    box[2] = int(box[2]) if box[2] <= width else width
    box[3] = int(box[3]) if box[3] <= height else height

    roi_h, roi_w = box[3]-box[1], box[2]-box[0]
    if roi_h < 1 or roi_w < 1:
        # 画像サイズがおかしい
        logger.warn('Incorrect roi image')
        return None
    return box

def get_top_k(predict_result, top_k):
    """
    識別結果からtop_kのデータを取得する
    """
    winner_score = np.sort(predict_result)[::-1]
    winner = np.argsort(predict_result)[::-1]
    return winner[:top_k].tolist(), winner_score[:top_k].tolist()
