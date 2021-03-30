## PyTorch-YOLOv3-Mask_Detection
● 使用YOLOv3实现口罩佩戴检测

环境：  
>Ubuntu 20.04  
>CUDA version：11.1  
>CUDNN version：8.0.4  
>Pytorch version：1.7.1  
>Python 3.8.5  

训练100个epochs，在验证集上效果：  
classes name |AP
-------------|----
face_not_mask|89.1  
face_mask    |98.2 

mAP：0.916

检测效果图：
![image](https://github.com/LY-Road/PyTorch-YOLOv3-Mask_Detection/blob/main/output/2C9EFCB40BE052BA6D556F206C9B9F67.png)
![image](https://github.com/LY-Road/PyTorch-YOLOv3-Mask_Detection/blob/main/output/64B20670984C4558227A18E15D333B24.png)
![image](https://github.com/LY-Road/PyTorch-YOLOv3-Mask_Detection/blob/main/output/6885980472858D01176A6A6C0FFB0A7D.png)
![image](https://github.com/LY-Road/PyTorch-YOLOv3-Mask_Detection/blob/main/output/93096EB90097F379ACF957169222CD67.png)


### 下载数据集
链接: https://pan.baidu.com/s/1SqjBKCO_IupeiCZYK3R73w 提取码: ftng
### 下载预训练权重
链接：https://pan.baidu.com/s/1mzt_o46gsa9_QllEAiVK4Q 提取码：7nx3  
将预训练权重放在weights文件夹下。
### 创建模型配置文件
> $ cd config/  
> $ bash create_custom_model.sh 2  
### 修改类别名  
将data/custom文件夹下的classes.names文件内容修改为：    
> face_not_mask  
> face_mask  
### 数据处理
   数据集解压后将JPEGImages中的图片放在data/custom/images下，Annotations文件夹放在data/custom/labels下，使用如下命令生成标签，（标签格式为[label_idx x_center y_center width height]）。
> $ cd data/  
> $ cd custom/  
> $ python3 label_processing.py  
### 划分训练集和测试集
使用如下命令生成train.txt和valid.txt：  
> $ cd data/  
> $ cd custom/  
> $ python3 make_train_valid.py   
### 训练  
执行如下命令（训练好的权重存放在checkpoints文件夹下）：
> $ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data  

加载预训练权重，执行如下命令：    
> $ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/yolov3_ckpt_99.pth
### 测试
执行如下命令：
> $ python3 test.py ----model_def config/yolov3-custom.cfg --data_config config/custom.data --weights_path weights/yolov3_ckpt_99.pth --class_path data/custom/classes.names  

注意权重路径。
### 检测
将需要检测的图片放在data/samples下，执行如下命令（输出图片存放在output文件夹下）：
> $ python3 detcet.py --model_def config/yolov3-custom.cfg --weights_path weights/yolov3_ckpt_99.pth --class_path data/custom/classes.names  

注意权重路径。

