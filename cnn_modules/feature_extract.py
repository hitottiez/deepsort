# coding:utf-8

import json
import traceback

import numpy as np

import cv2

from .ImageEncoder import ImageEncoder
from .util import tfsetup, validate_box


def decode_image(image_file):
    """
    画像をデコードする
    """
    nparr = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        print('Invalid Request Parameter[Decode `image` failed.]')
    return image

class CnnFeatureExtract(object):

    def __init__(self, weight_file):
        # モデル初期化
        tfsession, self.graph = tfsetup()
        self.encoder = ImageEncoder(weight_file, session=tfsession)

    def extract(self, image_file, param):
        """
        CNN特徴抽出API
        """
        try:
            # 画像パース
            image = decode_image(image_file)
            parameter = json.loads(param)
            # バウンディングボックスバリデーション
            boxes = list(map(lambda x: validate_box(image, x), parameter))
            # 有効なバウンディングボックスだけ抜き出す
            active_boxes = list(filter(lambda x: x is not None, boxes))
            # 特徴量抽出
            with self.graph.as_default():
                features = self.encoder.encode(image, active_boxes)
            return self._create_response(features, boxes)
        except:
            traceback.print_exc()
            print("CNN Error")
    
    def _create_response(self, features, boxes):
        """
        レスポンスデータ作成
        """
        results = []
        count = 0
        for box in boxes:
            if box is None:
                results.append({
                    'success': False,
                    'features': []
                })
            else:
                results.append({
                    'success': True,
                    'features': features[count].tolist()
                })
                count += 1
        return json.dumps(results)
