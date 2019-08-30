#Okutamaデータセットの行動認識結果評価(mAP評価)PG
# ※mAP評価はROADの評価プログラムを流用する
# ------------------------------------------------------------------------

import __addpath
import os
import argparse
import pickle
from tools.eval_utils.poc2_act_eval import evaluate_actions, all_video_eval
from tools.eval_utils.context_dataset import ContextDataSet
from tools.eval_utils.load_vatic_label import load_gt
from app.action_master import ActionMaster
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gt_root', help='ground truth directory. ex) /path/to/dataset/label/test')
parser.add_argument('--save_root', help='データセット識別結果ディレクトリ')
args = parser.parse_args()


def main():
    # -2はいらない
    master = ActionMaster('okutama')
    del master.__dict__()[-2]
    del master.__dict__()[-1]
    classes = list(master.__dict__().keys())
    classes = dict(zip(classes, range(len(classes))))

    logs = ContextDataSet(args.save_root)

    eval_results_of_file = {}
    df_data = {}
    row_list = list(master.__dict__().values())
    row_list.append('mAP')
    for video_name, logdata in logs.context_logs.items():

        print('Eval: {}'.format(video_name))
        gt = load_gt(
            args.gt_root, video_name, dataset_type='okutama')

        # ['1.1.9', '2.2.1', '1.1.8'] はフレーム数とgtの数が一致しない
        # （人がいなくなってから動画の最後までのフレームのgtがない
        # ので、スキップさせずに続ける。 
        if len(gt) != len(logdata) and video_name not in ['1.1.9', '2.2.1', '1.1.8']:
            assert len(gt) == len(logdata), '件数不一致'
        # 評価実行
        ap_all, eval_results = evaluate_actions(gt, logdata, classes)
        # 評価結果に行動ラベル情報を付与
        for cls_id, value in eval_results.items():
            value['label'] = master.findByModelID(cls_id)
        eval_results_of_file[video_name] = ap_all, eval_results

        df_data[video_name] = []
        for cls_id in classes.keys():
            if cls_id in eval_results.keys():
                df_data[video_name].append(eval_results[cls_id]['cls_ap'])
            else:
                df_data[video_name].append(0.0)

        df_data[video_name].append(ap_all)

        # 結果をpklで保存
        filepath = os.path.join(args.save_root, video_name, 'eval.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(eval_results_of_file[video_name], f)

    # 全ファイル合わせたmAP評価
    print('All Video Eval')
    print(args.save_root)
    ap_all, eval_results = all_video_eval(classes, eval_results_of_file)
    for cls_id, value in eval_results.items():
        value['label'] = master.findByModelID(cls_id)

    filepath = os.path.join(args.save_root, 'eval_all.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump((ap_all, eval_results), f)

    df_data['ALL'] = [d['cls_ap'] for d in eval_results.values()]
    df_data['ALL'].append(ap_all)
    df = pd.DataFrame(df_data, index=row_list)
    print('class average precision and mean average precision')
    print(df)

    print('Evaluation Complete')

if __name__ == '__main__':
    main()
