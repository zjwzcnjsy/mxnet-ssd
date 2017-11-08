## 人脸检测的实现

### 生成训练数据
``` bash
cd tools
python prepare_dataset.py --dataset wider --set train --root D:\res\face_detect\images\WIDER --target ../data/wider-train.lst
python prepare_dataset.py --dataset wider --set val --root D:\res\face_detect\images\WIDER --target ../data/wider-val.lst --shuffle False
```

### 训练
``` bash
cd ..
python train.py --train-path data/wider-train.lst --val-path data/wider-val.lst --gpus 0 --batch-size 32 --lr 0.004 --num-class 1 --num-example 12880 --class-names 'face' --voc07 False --label-width 9842
```