# deepsort
This repository includes modified code of [original deep_sort](https://github.com/nwojke/deep_sort) and it can utilize the primitive action features for human tracking.

## Cloning this repogitory and building a docker image
```
git clone --recursive https://github.com/hitottiez/deepsort.git
cd deepsort
docker build -t <tagname> .
```

## Running a docker container and login
Run:
```
docker run -d -it --name <container_name> \
    --mount type=bind,src=/<path/to/deepsort>/,dst=/opt/multi_actrecog/deepsort \
    --mount type=bind,src=/<path/to/dataset>/,dst=/mnt/dataset \
    <image name> /bin/bash
```

Login:
```
docker exec -it <container_name> /bin/bash
```

## Extraction of CNN features
Download models from [original deep_sort](https://github.com/nwojke/deep_sort).

Run:
```
python cnn_feature.py \
    --img_dir_path /mnt/dataset/okutama_action_dataset/okutama_3840_2160/images/  \
    --weight_file <path/to/mars-small128.ckpt-68577>
```

Then, `cnn.txt` is created in `/mnt/dataset/okutama_action_dataset/okutama_3840_2160/images/**/feature_results/`.

## Running human tracking and action recognition
Convert Okutama-Action dataset to split images using ffmpeg and put feature files (`det.txt`, `cnn.txt`, `{rgb, flow, fusion}_.txt`) in the proper directory (refer [mht-paf](https://github.com/hitottiez/mht-paf)).

Example in case that dataset is `/mnt/dataset/okutama_action_dataset/okutama_3840_2160/` and the file of primitive action feature is `rgb_tsn.txt`:
```
python batch_okutama.py \
    --data_root /mnt/dataset/okutama_action_dataset/okutama_3840_2160/ \
    --save_root /mnt/dataset/okutama_action_dataset/deepsort_tracking_result/ \
    --tsn_modality rgb \
    --worker 5
```
When not using the primitive action feature, set `enable_tsn_matching` in `config/multi_actrecog.ini` as `false`.
(as same as original deep_sort)

Then, `contextlog.dat` is created in `/mnt/dataset/okutama_action_dataset/deepsort_tracking_result` as the following directory structure:
```
/mnt/dataset/okutama_action_dataset/deepsort_tracking_result/
├── 1.1.8
│   └── contextlog.dat
├── 1.1.9
├── 1.2.1
├── 1.2.10
├── 1.2.3
├── 2.1.8
├── 2.1.9
├── 2.2.1
├── 2.2.10
└── 2.2.3
```

## Evaluation
### Human tracking (Recall, Precision, ID switch, MOTA, ...)
Set `--save_root` as the directory where the tracking results are stored and run:
```
cd tools
python okutama_moteval.py \
    --gt_root /mnt/dataset/okutama_action_dataset/okutama_3840_2160/labels/test/ \
    --save_root /mnt/dataset/okutama_action_dataset/deepsort_tracking_result/
```

Then, the evaluation results are displayed in the console and saved in `motmetric.csv`.

### Action detection (mAP)
Run:
```
cd tools
python okutama_acteval.py \
    --gt_root /mnt/dataset/okutama_action_dataset/okutama_3840_2160/multi_labels/test/ \
    --save_root /mnt/dataset/okutama_action_dataset/deepsort_tracking_result/
```
Note that `gt_root` need to be set as the multilabel directory.

The evaluation results are saved in `eval_all.csv`.
```
                   1.2.10     1.1.9         2.2.1     1.2.3    2.2.10     2.1.8     2.2.3     1.1.8     1.2.1     2.1.9       ALL
Calling          0.007968  0.000000  0.000000e+00  0.000000  0.000000  0.000000  0.000000  0.000000  0.003546  0.000000  0.002530
Carrying         0.212147  0.000287  0.000000e+00  0.000300  0.000059  0.000002  0.000000  0.014212  0.000050  0.000201  0.023015
Drinking         0.000000  0.000000  0.000000e+00  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
Hand Shaking     0.000000  0.018750  0.000000e+00  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.094341  0.010858
Hugging          0.043939  0.000000  0.000000e+00  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.010385
Lying            0.005291  0.000000  0.000000e+00  0.000000  0.032661  0.000000  0.000000  0.000000  0.000000  0.000000  0.017957
Pushing/Pulling  0.352467  0.030965  6.696470e-05  0.020916  0.029440  0.265301  0.000000  0.000000  0.261937  0.067376  0.139742
Reading          0.017274  0.007913  0.000000e+00  0.036591  0.000000  0.002551  0.000000  0.004684  0.000000  0.000000  0.003894
Running          0.000000  0.000000  0.000000e+00  0.021709  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.016083
Sitting          0.093303  0.007168  3.890931e-02  0.112873  0.002075  0.139750  0.032064  0.008299  0.087836  0.056599  0.047960
Standing         0.099241  0.000111  8.846145e-07  0.003353  0.001166  0.005629  0.011065  0.005461  0.022282  0.009014  0.005273
Walking          0.198312  0.001178  1.675798e-04  0.060860  0.003006  0.047032  0.000105  0.014918  0.091606  0.037590  0.052454
mAP              0.114438  0.007375  5.592106e-03  0.028511  0.007601  0.051141  0.004804  0.004757  0.051917  0.033140  0.027513
```
