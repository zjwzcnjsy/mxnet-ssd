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
python train.py --train-path data/wider-train.rec --val-path data/wider-val.rec --prefix model/ssd-wider --gpus 0,1,2,3 --batch-size 32 --lr 0.001 --num-class 1 --num-example 12880 --class-names 'face' --voc07 False --label-width 9842
```

### 测试
生成检测版本的模型：
``` bash
python deploy.py --network vgg16_reduced --epoch 139 --prefix model/ssd-wider --data-shape 300 --num-class 1
```

测试生成的模型：
``` bash
python demo.py --deploy --epoch 139 --prefix model/deploy_ssd-wider --gpu 0 --data-shape 300 --class-names 'face' --images data/demo/000001.jpg
```