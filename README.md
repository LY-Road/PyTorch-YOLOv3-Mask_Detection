## PyTorch-YOLOv3-Mask_Detection
● 口罩佩戴检测

环境：  
>Ubuntu 20.04  
>CUDA version：11.1  
>CUDNN version：8.0.4  
>Pytorch version：1.7.1  
>Python 3.8.5  

### 下载数据集
链接: https://pan.baidu.com/s/1SqjBKCO_IupeiCZYK3R73w 提取码: ftng
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
执行如下命令：
> $ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data
### 测试
执行如下命令：
> python3 test.py --
### 检测
将需要检测的图片放在data/samples下，执行如下命令：
> python3
