captcha_pytorch深度学习识别验证码
=========

本项目致力于使用神经网络来识别验证码。

特性
===
- __验证码包括数字、大写字母、小写字母__
- __四位数字 + 大写字符 + 小写，验证码识别率约 95 %__


原理
===

- __训练卷积神经网络__
    自定义构建一个多层的卷积网络，进行多标签分类模型的训练
    标记的每个字符都做 one-hot 编码



快速开始
====
- __一：安装环境__

    - Python3.7
    - Pytorch(参考官网http://pytorch.org)
    - torchvision
    - cuda(训练时需要)
    - 预训练的模型可以+作者微信


- __二：训练模型__
    ```bash
    python train.py
    ```
    执行以上命令，会读取目录 ```/train/train_label.csv``` 根据ID和label读取数据，使用```tain_model.py```训练 ```/train``` 文件夹下的图片，最终训练完成会生成在 ```/model``` 目录里生成 */{**model_name**}_epoch{**epoch**}_loss{**loss.item**}_acc_{**acc**}.pkl*
    
- __三：模型预测__
    ```bash
    python test.py
    ```
    使用（二）生成的模型进行预测 ```\test``` 文件夹里的所有图片，并在根目录生成 ```resul.csv```

- __三：修改模型__
    在```tain_model.py```修改，自定义修改后进行不同的实验训练


作者
===
* __Ractive__ <chengu.ractive@gmail.com> **wx:ractive89**


声明
===
本项目仅用于交流学习