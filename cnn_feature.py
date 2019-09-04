# coding:utf-8

import argparse
from cnn_modules.common_processing_class import Common
from cnn_modules.feature_extract import CnnFeatureExtract


def main():
    # 引数まわり
    parser = argparse.ArgumentParser(description='CNN特徴量作成ツール')
    parser.add_argument('--img_dir_path', help='/path/to/dataset/images')
    parser.add_argument('--weight_file', help='重みファイルを指定')
    args = parser.parse_args()

    cnn_extractor = CnnFeatureExtract(args.weight_file)

    common_instance = Common(img_dir_path=args.img_dir_path,
                             model_type='CNN',
                             predict_function=cnn_extractor.extract)
    common_instance.bbox_file_loop()


if __name__ == '__main__':
    main()
