## PyTorch-YOLOv3-Mask_Detection
● 人脸口罩佩戴检测

>环境：  
>Ubuntu 20.04  
>CUDA version：11.1  
>CUDNN version：8.0.4  
>Pytorch version：1.7.1  
>Python 3.8.5  

### 1.下载数据集
链接: https://pan.baidu.com/s/1SqjBKCO_IupeiCZYK3R73w 提取码: ftng
### 2.数据处理
   数据集解压后将JPEGImages中的文件放在data/custom/images下，Annotations中的文件放在data/custom/labels下，使用如下命令转换标签格式( (x_min, y_min, x_max, y_max) -->  (x, y, w, h) )。  
$ cd data/  
$ cd custom/  
$ python3 label_processing.py
### 3.创建model config
