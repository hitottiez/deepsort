
""" Evaluation code based on VOC protocol

Original author: Ellis Brown, Max deGroot for VOC dataset
https://github.com/amdegroot/ssd.pytorch

Updated by Gurkirt Singh for ucf101-24 dataset

"""

import os
import numpy as np


def voc_ap(rec, prec, use_07_metric=False):
    """
    VOC評価(ROADから抜粋)
    ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # print('voc_ap() - use_07_metric:=' + str(use_07_metric))
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_iou(cls_gt_boxes, box):
    """
    IoUマッチング(ROADから抜粋)
    """
    ious = np.zeros(cls_gt_boxes.shape[0])

    for m in range(ious.shape[0]):
        gtbox = cls_gt_boxes[m]

        xmin = max(gtbox[0], box[0])
        ymin = max(gtbox[1], box[1])
        xmax = min(gtbox[2], box[2])
        ymax = min(gtbox[3], box[3])
        iw = np.maximum(xmax - xmin, 0.)
        ih = np.maximum(ymax - ymin, 0.)
        if iw > 0 and ih > 0:
            intsc = iw*ih
        else:
            intsc = 0.0
        # print (intsc)
        union = (gtbox[2] - gtbox[0]) * (gtbox[3] - gtbox[1]) + \
            (box[2] - box[0]) * (box[3] - box[1]) - intsc
        ious[m] = intsc/union

    return ious


def generate_det_boxes(det_contexts, classes):
    """
    検出データを処理しやすいように持ち直し
    """
    num_frames = len(det_contexts)
    dataset = [[] for _ in range(num_frames)]
    for frame_idx, context in enumerate(det_contexts):
        for det in context['context']:
            action_id = det['action']['model_id']
            score = det['action']['score']
            box = det['location']
            bbox = np.asarray(box + score, dtype=np.float32)
            dataset[frame_idx].append({
                'bbox': bbox,
                'action_id': action_id
            })
    return dataset


def evaluate_actions(gt, det_contexts, classes, iou_thresh=0.5):
    """
    動画単位の評価結果を取得する
    """
    num_positives = 0 # 正解の数
    num_frames = len(gt)
    num_classes = len(classes)
    det_boxes = generate_det_boxes(det_contexts, classes)

    ap_all = np.array([], dtype=np.float32)
    scores_of_cls = np.zeros((num_classes, num_frames * 220)) # 行動認識結果スコアログ
    istps = np.zeros((num_classes, num_frames * 220)) # True-Positiveログ
    det_counts_of_cls = np.zeros(num_classes, dtype=np.int) # 行動ごとの検出結果
    num_positives_of_cls = np.zeros(num_classes, dtype=np.int)  # 行動ごとの正例数

    for nf in range(num_frames): # フレーム単位
        gt_data = gt[nf] # 当該フレームの正解データ
        boxes = sorted(det_boxes[nf], key=lambda x: -x['bbox'][-4])  # 当該フレームの検出結果をスコア順に並べる
        gt_actions_of_boxes = [x['action_id'] for x in gt_data]  # 当該フレームの、各ボックスに割り当てられている正解ラベルセット(e.g. [[1, 2], [1], ...])
        gt_boxes = np.asarray([x['box'] for x in gt_data], dtype=np.float32)  # 当該フレームのバウンディングボックス

        # 正例カウント
        for gt_ids in gt_actions_of_boxes:
            gt_indices = [classes[x] for x in gt_ids if x != -1]  # exclude No Action
            num_positives_of_cls[gt_indices] += 1

        # 検出結果評価
        if len(boxes) > 0:
            for box_data in boxes:  # 検出結果単位で繰り返し
                box = box_data['bbox'][:-4]    # バウンディング
                score = box_data['bbox'][-4:]   # スコア
                act_ids = box_data['action_id']  # 行動ID

                # 全てNoActionの場合は一個のみ。
                # それ以外でNoActionが含まれる場合は、除外する（他の行動が含まれているので必要ない）
                if act_ids == [-1, -1, -1, -1]:
                    act_ids = [-1]
                    score = [score[0]]
                else:
                    act_ids = [i for i in act_ids if i != -1]
                    score = [score[i]
                             for i, act_id in enumerate(act_ids)
                             if act_id != -1]

                # 更新する対象の行動セット(デフォルトは検出した行動のID)
                target_dict = {idx: classes[i] if i != -1 else -1
                               for idx, i in enumerate(act_ids)}  # 行動IDに紐づくインデックス

                is_positive_dict = {
                    idx: False for idx, _ in enumerate(act_ids)}

                # 正解ラベルとマッチング
                if gt_boxes.shape[0] > 0:  # マッチしていない正解データがある正解データがある
                    iou = compute_iou(gt_boxes, box)  # IoU計算
                    maxid = np.argmax(iou)  # 最もマッチしたバウンディングボックスを取り出す
                    gt_ids = gt_actions_of_boxes[maxid]

                    if iou[maxid] >= iou_thresh:
                        # 正解データがNoActionならdetection,gt共に評価対象外とする
                        if - 1 in gt_ids:
                            gt_boxes = np.delete(gt_boxes, maxid, 0)
                            del gt_actions_of_boxes[maxid]
                            continue

                        for top_i, act_id in enumerate(act_ids):
                            is_positive_dict[top_i] = act_id in gt_ids

                        # 検出の行動数は、okutama-action datasetの最大同時ラベル数に合わせて、
                        # 常に4つの行動ラベルを出力している。
                        # そのため正解の行動数よりも検出の行動数のほうが多くなるケースが出てくる。
                        # それをdet_counts_of_clsに含めるのはよろしくないので、この場合は
                        # 検出の行動数を正解の行動数と一致するよう削る
                        # ではどれを削るか？
                        #   正解とマッチしてないもののうち、スコアの低いものから削る。
                        if len(gt_ids) < len(is_positive_dict):
                            fp_list = [
                                act_id for act_id, is_positive
                                in list(is_positive_dict.items())
                                if is_positive is False
                            ]
                            while (len(gt_ids) != len(is_positive_dict)):
                                min_scored_fp_act_id = fp_list.pop(-1)
                                del is_positive_dict[min_scored_fp_act_id]
                                del target_dict[min_scored_fp_act_id]

                        # 正解データからマッチしたデータを削除
                        gt_boxes = np.delete(gt_boxes, maxid, 0)
                        del gt_actions_of_boxes[maxid]

                # istp、score、det_count更新
                # 正解した場合: 正解に設定されているラベル全てを更新
                # 外した場合: 検出したものだけ検出結果更新
                for top_i, class_value in target_dict.items():
                    if class_value == -1:
                        continue
                    det_count = det_counts_of_cls[class_value]  # 当該行動の現在の検出件数
                    scores_of_cls[class_value, det_count] = score[top_i]
                    if is_positive_dict[top_i] is True:
                        istps[class_value, det_count] = 1
                    det_counts_of_cls[class_value] += 1

    # 評価実行
    eval_results = {}
    for cls_id, cls_ind in classes.items():
        det_count = det_counts_of_cls[cls_ind]
        num_positives = num_positives_of_cls[cls_ind] if num_positives_of_cls[cls_ind] > 0 else 1
        if num_positives_of_cls[cls_ind] == 0:
            print('no gt of action', cls_id)
            continue
        
        scores = scores_of_cls[cls_ind, :det_count]
        istp = istps[cls_ind, :det_count]
        argsort_scores = np.argsort(-scores)
        istp = istp[argsort_scores]
        fp = np.cumsum(istp == 0) # fp[-1] ==>  誤検出(行動が違う)
        tp = np.cumsum(istp == 1) # tp[-1] ==> 正解を正解とした
        fp = fp.astype(np.float64)
        tp = tp.astype(np.float64)
        recall = tp / float(num_positives)  # compute recall
        # compute precision
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        cls_ap = voc_ap(recall, precision)
        eval_results[cls_id] = {
            'num_positives': num_positives_of_cls[cls_ind],
            'fp': fp,
            'tp': tp,
            'recall': recall,
            'precision': precision,
            'istp': istps[cls_ind, :det_count],
            'scores': scores_of_cls[cls_ind, :det_count],
            'cls_ap': cls_ap
        }
        ap_all = np.append(ap_all, cls_ap)
    return np.mean(ap_all), eval_results


def all_video_eval(classes, eval_results_of_file):
    num_classes = len(classes)
    scores_of_cls = [[] for _ in range(num_classes)]
    istp_of_cls = [[] for _ in range(num_classes)]
    ap_all = np.zeros(num_classes, dtype=np.float32)
    num_positives_of_cls = np.zeros(num_classes, dtype=np.int)
    for _, (_, eval_results) in eval_results_of_file.items():
        for cls_id, eval_data in eval_results.items():
            cls_idx = classes[cls_id]
            scores = eval_data['scores']
            istp = eval_data['istp']
            num_positives = eval_data['num_positives']
            num_positives_of_cls[cls_idx] += num_positives
            scores_of_cls[cls_idx].append(scores)
            istp_of_cls[cls_idx].append(istp)
    results = {}
    for cls_id, cls_ind in classes.items():
        scores = np.concatenate(scores_of_cls[cls_ind])
        istp = np.concatenate(istp_of_cls[cls_ind])
        num_positives = num_positives_of_cls[cls_ind]
        if num_positives < 1:
            num_positives = 1
        argsort_scores = np.argsort(-scores)
        istp = istp[argsort_scores]
        fp = np.cumsum(istp == 0)  # fp[-1] ==>  誤検出(行動が違う)
        tp = np.cumsum(istp == 1)  # tp[-1] ==> 正解を正解とした
        fp = fp.astype(np.float64)
        tp = tp.astype(np.float64)
        recall = tp / float(num_positives)  # compute recall
        # compute precision
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        cls_ap = voc_ap(recall, precision)
        ap_all[cls_ind] = cls_ap
        results[cls_id] = {
            'num_positives': num_positives_of_cls[cls_ind],
            'fp': fp,
            'tp': tp,
            'recall': recall,
            'precision': precision,
            'istp': istp,
            'scores': scores,
            'cls_ap': cls_ap
        }
    return np.mean(ap_all), results
