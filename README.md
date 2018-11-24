# Deep Learning-Based Crack Damage Detection Using Convolutional Neural Networks
> [Cha_et_al-2017-Computer-Aided_Civil_and_Infrastructure_Engineering.pdf]()
## 工具,以及资源

1. tensorflow<p>
2. PIL<p>
3. numpy<p>
4. 数据:<https://data.mendeley.com/datasets/5y9wdsg2zt/1>

## 网络结构 [model.py](https://gitee.com/onexming/Notebook/blob/master/Concrete-Crack-Detection/model.py)<p>
![![网络结构]()<P>](https://gitee.com/uploads/images/2018/0614/201638_baa89c6a_1498089.png "
![我的网络结构]()屏幕截图.png")
![![网络细节]()<p>](https://gitee.com/uploads/images/2018/0614/201725_175e2eb2_1498089.png "屏幕截图.png")

### 超参数
1. learn rate: 1e-4
1. learn decay: 0.99
1. batch normalization momentum: 0.9
1. dropout rate: 0.5
1. Optimizer: SGD/adam<p>
使用adam优化是速度快于SGD
## 数据处理  [raw_to_tfrecoder.py]()
将图片数据处理为tfrecoder, 将大图处理为相同格式的小图,<p>
将小图送入模型中得出值, 根据返回值画出大图中损坏的位置
## 测试函数  [evalution.py](https://gitee.com/onexming/Notebook/blob/master/Concrete-Crack-Detection/raw_to_tfrecoder.py)<p>
正确率
![输入图片说明](https://gitee.com/uploads/images/2018/0614/202307_e6c9f036_1498089.png "屏幕截图.png")
loss
![loss](https://gitee.com/uploads/images/2018/0614/202352_cedab2a5_1498089.png "屏幕截图.png")

### 运行效果 
在训练图片中缝隙的颜色大多数为黑色, 在缝隙颜色和墙面颜色区分比较明显的时候,识别效果较好<p>
![![输入图片说明](https://gitee.com/uploads/images/2018/0614/202515_d2970f0d_1498089.png "屏幕截图.png")](https://gitee.com/uploads/images/2018/0614/202514_92d160eb_1498089.png "屏幕截图.png")
![输入图片说明](https://gitee.com/uploads/images/2018/0614/202522_12c13990_1498089.png "屏幕截图.png")
在墙面颜色为比较暗的颜色时, 与缝隙的区分度不是那么高时,识别效果明显变差<p>
![输入图片说明](https://gitee.com/uploads/images/2018/0614/202538_6a47472a_1498089.png "屏幕截图.png")
![输入图片说明](https://gitee.com/uploads/images/2018/0614/202548_b52aabbc_1498089.png "屏幕截图.png")
![输入图片说明](https://gitee.com/uploads/images/2018/0614/202556_869f5f46_1498089.png "屏幕截图.png")