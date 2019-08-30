# PoC2データセットの推論をバッチ実行する
# -------------------------------------------------------------------------
from app import config
import os
import argparse
from run_core_engine import main as run_app
from multiprocessing.pool import Pool
import json

# 引数パラメータ初期設定
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', help='okutama-Action dataset dir')
parser.add_argument('--save_root', help='save result dir')
parser.add_argument('--tsn_modality', choices=['rgb', 'flow', 'fusion'])
parser.add_argument('--worker', type=int, default=2, help='worker num')
parser.add_argument('--target', choices=['train', 'test'], default='test', help='どのデータセットを対象とするか')

root_args = parser.parse_args()
root_args.dataset_type = 'okutama'

def save_config(save_root):
    ini_filepath = os.path.join(save_root, 'settings.ini')
    with open(ini_filepath, 'w') as f:
        config.ini.write(f)

def save_args(args):
    jsondata = json.dumps(vars(args), indent=4)
    json_filepath = os.path.join(args.save_root, 'args.json')
    with open(json_filepath, 'w') as f:
        f.write(jsondata)

if __name__ == '__main__':
    rgb_dir = os.path.join(root_args.data_root, 'images')
    label_dir = os.path.join(root_args.data_root, 'labels', root_args.target)
    # save_rootディレクトリがなければ作成する
    if not os.path.exists(root_args.save_root):
        os.makedirs(root_args.save_root)

    # 全データの出力パラメータ作成
    mp_args = []
    for label_file in sorted(os.listdir(label_dir)):
        args = argparse.Namespace(**vars(root_args))
        filename, _ = os.path.splitext(label_file)
        args.input = os.path.join(rgb_dir, filename)
        args.output = os.path.join(args.save_root, filename)
        mp_args.append(args)

    save_config(root_args.save_root)
    save_args(root_args)

    with Pool(root_args.worker) as p:
        p.map(run_app, mp_args)
