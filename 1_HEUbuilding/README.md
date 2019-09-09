# 深度学习图像识别实验
> 2019年9月份, HEU楼宇识别实验

- 本次实验为一个多标签分类任务（Multi-label classification）。其中，输入图像为车载摄像头拍摄的哈尔滨工程大学道路与学校楼宇。输出类别一共有：class labels: 1.空类 2. 21Front 3. 21West 4. 51Front 5. 61East 6. 61Inside 7. 61North 8. 61Park 9. 61South 10. 61West 11. Dormi13Back 12. Dormi13Front 13. Dormi13North 14. Dormi1617Front 15. Dormi3Back 16. Dormi4 17. Dormi4Road 18. DormiTen 19. ForeiDormiSouth 20. Hospital 21. InternationCenter 22. Lib 23. MainBack

- 本次实验应用的神经网络
  - 本次神经网络使用的是smaller VGG 卷积神经网络，具体网络参数如下：
第一层：卷积层，卷积核3x3
第二层：池化层，
第三层：卷积层，卷积核3x3
第四层：池化层，
第五层：卷积层，卷积核3x3
第六层：池化层
第七层：卷积层，卷积核3x3
第八层：池化层，
第九层：全连接层，1024个神经元
第十层：全连接层，神经元数量为总类别数23

  - 训练过程中，采用了dropout技术，防止模型过拟合。

- 输入图像原图像维度为（1080, 1920, 3），图像预处理转换为(192, 108) ，输入神经网络，输出神经元。

- 训练过程，总回合数为100，训练精准度达到98%左右。

![](https://github.com/MorganWoods/Deep_Learning/blob/master/1_HEUbuilding/src/plot.png)

- 本模型使用方法
  - 在python3环境下使用,需要安装的库有:Tensorflow, Keras, numpy, skimage, cv2, 等;  
  - 下载model/整个文件夹,并新建testpic/ 子文件夹,将需要测试的图片放入这个文件夹中.
  - 在电脑命令行模式下,python3工作环境,使用命令 ```python prediction.py -n=name```  ;  其中name为图片的名字,不含扩展名. 命令行随后会输出预测的类别与他的概率.
  
- 样例
![](https://github.com/MorganWoods/Deep_Learning/blob/master/1_HEUbuilding/src/sample1.png)
![](https://github.com/MorganWoods/Deep_Learning/blob/master/1_HEUbuilding/src/sample2.png)
![](https://github.com/MorganWoods/Deep_Learning/blob/master/1_HEUbuilding/src/sample3.png)
  
