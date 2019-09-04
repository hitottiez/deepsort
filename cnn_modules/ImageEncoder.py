# ImageEncoder
# ==============================================================================

from vendors.deep_sort.generate_detections import create_box_encoder, extract_image_patch
import numpy as np


class ImageEncoder:
    """
    CNN Image Encoder
    """

    def __init__(self, model_filename, batch_size=32, session=None):

        self.encoder = create_box_encoder(model_filename,
                                          batch_size=batch_size)

    def encode(self, image, boxes):
        """
        画像をエンコードして128次元の特徴データに変換する
        :param image: OpenCV画像データ(BGR色空間)
        :param boxes: バウンディングボックスリスト
                      * numpy形式、 Nx4 の行列データ
                      * バウンディングボックスは[左上x, 左上y, 右下x, 右下y]
        """
        if len(boxes) > 0:
            # 左上x, 左上y, width, heightに変換する
            boxes = [[box[0], box[1], box[2]-box[0], box[3]-box[1]]
                     for box in boxes]
            return self.encoder(image, np.array(boxes))
        else:
            return np.array([])