# 1 MTCNN-tensorflow
人脸检测MTCNN算法，采用tensorflow框架编写，代码中文注释完整。
有关MTCNN算法理论可以参考我的博客：[MTCNN算法笔记](https://blog.csdn.net/weixin_41965898/article/details/84589666)

# 2 训练集说明
- 人脸检测 WIDERFace[数据下载](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
WIDER FACE数据集是人脸检测基准数据集，其中的图像是从公开的WIDER数据集中选择的。选择32,203个图像并标记393,703个面部，数据集基于61个事件类进行组织。对于每个事件类，我们随机选择40％/ 10％/ 50％的数据作为训练，验证和测试集。
训练集12880个图像，已下载到`train_data/Wider_train`中，其标签文件有如下两种。

标记文件1为`train_data/wider_face_train.txt`，格式为：
```
0--Parade/0_Parade_Parade_0_1014 121.69 379.67 131.92 391.39 245.55 378.44 257.67 392.15 
**
第一个数据为文件名：0--Parade/0_Parade_Parade_0_1014.jpg 
接下来没四个数据一组，表示一个人脸的BOX：121.69 379.67 131.92 391.39 为第一张脸，245.55 378.44 257.67 392.15 为第二张脸。
**
```
标记文件2为`train_data/wider_face_train_bbx_gt.txt`，格式为：
```
0--Parade/0_Parade_marchingband_1_117.jpg
9
69 359 50 36 1 0 0 0 0 1 
227 382 56 43 1 0 1 0 0 1 
296 305 44 26 1 0 0 0 0 1 
353 280 40 36 2 0 0 0 2 1 
885 377 63 41 1 0 0 0 0 1 
819 391 34 43 2 0 0 0 1 0 
727 342 37 31 2 0 0 0 0 1 
598 246 33 29 2 0 0 0 0 1 
740 308 45 33 1 0 0 0 2 1 
**
文件名：0--Parade/0_Parade_marchingband_1_117.jpg

标记框的数量：9

边界框：x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose

其中 x1,y1 为标记框左上角的坐标，w，h 为标记框的宽度

blur, expression, illumination, invalid, occlusion, pose 为标记框的属性，表示是否模糊，表情，光照情况，是否有效，是否遮挡，姿势 

**
```
- 人脸关键点（5个关键点）[数据下载](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)
该训练集包含5,590 LFW images 和 7,876 other images
其5590张LFW照片已下载到`train_data/lfw_5590`中。
标记文件为`train_data/trainImageList.txt`，格式为：

```

lfw_5590\Abbas_Kiarostami_0001.jpg 75 165 87 177 106.750000 108.250000 143.750000 108.750000 131.250000 127.250000 106.250000 155.250000 142.750000 155.250000

**
第一个数据为文件名：lfw_5590\Abbas_Kiarostami_0001.jpg
第二和第三个数据为标记框左上角坐标：  75 165 
第四和第五个数据为标记框长宽：  87 177
第六和第七个数据为左眼标记点：106.750000 108.250000 
第八和第九个数据为右眼标记点：143.750000 108.750000 
第十和第十一个数据为鼻子标记点：131.250000 127.250000
第十二和第十三个数据为左嘴标记点：106.250000 155.250000 
第十四和第十五个数据为右嘴标记点：142.750000 155.250000
**
```

# 3 文件、目录结构
## 3.1 train_data/
### 从WiderFace训练集生成样本，用于人脸分类和边框回归
- gen_12net_data.py
    提供PNet的数据采样
- gen_hard_example.py
    分别生成RNet和ONet的训练数据，hard_example代表挑选前 70% 较大的损失（基于前一个网络的预测结果）对应的样本作为困难样本
- data_utils.py
    从`wider_face_train_bbx_gt.txt`读取lable给gen_hard_example.py
### 从LFW-5590训练集生成样本，用于边框回归和关键点检测
- gen_landmark_tfrecords_aug_xx.py
    用于生成特征点的数据，在这里并没有生成tfreord,只是对进行数据增强（随机镜像、随机旋转）
    此脚本的输入是trainImageList.txt,其中定义了文件的路径，人脸框的位置（x1,x2,y1,y2）,特征点的位置（x1,y1,,,,,x5,y5）
- BBox_utils.py/Landmark_utils.py 
    提供函数给gen_landmark_tfrecords_aug_xx.py，用于特征点和边框处理，如随机旋转一个角度

### 合并样本集的lable
- gen_imglist_xxxnet.py
    分别将三个网络的三个任务（分类，回归，特征点检测）的lable汇总到一个文件中

### 样本数据和lable转为tfrecord
- gen_xx_tfrecords.py
    分别生成3个网络的tfrecord,在这里需要注意：
     -- PNet的训练数据(pos,neg,part,landmark)是混在一起的，生成了一个tfrecord，整合规则是part 和pos中各随机取250000条，neg随机取750000条（不足就全取），landmark全取
     -- RNet和ONet的各自需要生成4个tfrecord(pos,neg,part,landmark)，因为要控制各部分的样本比例（1：3：1：1）
- tfrecord_utils.py
 提供函数给gen_xx_tfrecords.py，读入训练的12x12等图片到tfrecord文件

### 工具
- utils.py
    提供IoU计算，提供将推荐框转为正方形
- loader.py
    迭代器，用于读取图片，功能其一是给训练打batch后提供batch图片，其二是用于测试读取单张图片（one_image_test.py)
- minibatch.py
    将读取到图片封装成一个batch_size大小，提供这些函数给loader.py
- nms.py
    提供非极大值抑制计算给gen_hard_example.py，合并前一个网络的重复推荐框

## 3.2 train_models/

- mtcnn_model.py

 PNet、RNet和ONet的CNN模型定义处

- train.py

 模型的训练代码，sess.run(net,数据)

- train_xxxNet.py

训练PNet,RNet,ONet的入口文件，可以修改参数，如学习率，

- read_tfrecord_v2.py /tfrecord_utils.py
    用于读取tfrecord数据，并对其解析
- MTCNN_config.py

 一些参数的配置，如BATCH_SIZE


## 3.3 Detection/
### 训练网络
- fcn_detector.py
 定义pnet人脸检测器，主要用于PNet的单张图片识别
- detector.py
 定义rnet和onet人脸检测器，用于RNet和ONet的一张图片通过PNet截取的多个人脸框的批次识别
- MtcnnDetecor.py
 将三个检测器汇集在一起，非极大值抑制取出重叠窗口，识别人脸和生成RNet，ONet输入数据

## 3.4 test/

- one_image_test.py

用于测试模型



## 3.5 data/MTCNN_model

训练好的模型保存处


# 4 训练
模型主要通过PNet，RNet，ONet三个网络级联，一步一步精调来对人脸进行更准确的检测。

**三个模型要按顺序训练，首先是PNet,然后RNet,最后ONet。**

## 4.1 PNet数据产生
**PNet数据是怎么来的呢？**

![image](https://github.com/luckyluckydadada/MTCNN_tf/blob/master/iou.png)

训练数据由四部分组成：pos,part,neg,landmark，比例为1：1：3：1。
**数据：**这四种图像都resize成12x12作为PNet的输入。

**label：**

pos、part的label：含有它们的类别1、-1；还有人脸真实框相对于截取窗口左上角的偏移量（正值代表向右和向下），偏移量除以截取窗口大小做了归一化，截取窗口为size x size大小， **size=npr.randint(int(0.8min(w,h)),np.ceil(1.25max(w,h)))**；

neg的label：只含有类别0；

landmark的label：含有类别-2；5个关键点的坐标偏移也是进行了归一化的。

- 进入`triain_data/`目录
- 运行 `gen_12net_data.py PNet`
从WiderFace训练集`train_data/Wider_train`中生成 12*12 的 positive人脸、negative人脸、part(IOU =0.4到0.65)人脸图片和lable文件
图片输出到`train_data/12`中的`positive、negative、part`目录，lable写入`train_data/12`中的`neg_12.txt、pos_12.txt、part_12.txt`
```
生成的`neg_12.txt`格式为：
12/negative/0.jpg 0                                    #0 表示negative
生成的`pos_12.txt`格式为：
12/part/0.jpg -1 0.23 0.02 -0.09 -0.15        # -1 表示part  
生成的`part_12.txt`格式为：
12/positive/0.jpg 1 0.05 -0.10 -0.10 -0.05   # 1 表示positive

0.23 0.02 -0.09 -0.15 和 0.05 -0.10 -0.10 -0.05表示如下的offset_x1、offset_y1、offset_x2、offset_y2：
#人脸框相对于截取图片的偏移，并做归一化处理           
            offset_x1=(x1-nx1)/float(size)
            offset_y1=(y1-ny1)/float(size)
            offset_x2=(x2-nx2)/float(size)
            offset_y2=(y2-ny2)/float(size)
```

- 运行 `gen_landmark_aug_12.py` 
对于LFW-5590训练集`train_data/lfw_5590`，**对每个LFW每一个图，遍历十次，随机产生一些box**，如果这个box和正确box的IOU > 0.65才保留。
之后对这个box进行适当的翻转，获得新的landmark和剪裁图片。
新的landmark进入`train_data/12/landmark_12_aug.txt`文件，剪裁后的图片进入`train_data/12/train_PNet_landmark_aug`目录。
```
#生成的`landmark_12_aug.txt`文件格式：
12/train_PNet_landmark_aug\0.jpg -2 0.288961038961039 0.20454545454545456 0.814935064935065 0.262987012987013 0.5357142857142857 0.6590909090909091 0.275974025974026 0.8538961038961039 0.724025974025974 0.9058441558441559
各字段含义：
12/train_PNet_landmark_aug\0.jpg：剪裁后的图片名称train_data\imglists\PNet
-2：代表LFW训练集得到的IOU > 0.65的positive图片，对应上面WiderFace中的0、1 、-1表示的negative、positive、part，我们把这类landmark数据用-2代表
0.288961038961039  。。。0.9058441558441559： 每两个一组，表示剪裁后的图片中人脸关键点，是归一后的值，不含边界框的值
```

- 运行 `gen_imglist_pnet.py`
整合 pos + neg + part + landmark，将`train_data/12`目录中`neg_12.txt、pos_12.txt、part_12.txt`和`landmark_12_aug.txt`整合到`train_data\imglists\PNet\train_PNet_landmark.txt`。
整合规则是part 和pos中各随机取250000条，neg随机取750000条（不足就全取），landmark全取。

- 运行 `gen_PNet_tfrecords.py`
首先将`imglists/PNet/train_PNet_landmark.txt`文件导入python列表，**negative的box默认为0，part和positive的box只包含人脸框，landmark的box只包含关键点，类别分别为0、-1、1、-2**；
其次，读入`triain_data/12/`生成的图片；
最后将列表中数据random.shuffle（随机打乱顺序），利用tf.python_io.TFRecordWriter将列表中的数据以TFRecord的形式写入磁盘文件`imglists/PNet/train_PNet_landmark.tfrecord_shuffle`中。
**注意：**
这些tfrecord数据包含lable文件（是图片的框、关键点、类别），和**训练数据（12x12图片）**。

## 4.2 训练PNet
**PNET四种不同的数据该怎么训练呢？**

这四种图像都resize成12x12作为PNet的输入，通过PNet得到了是否有人脸的概率[batch,2]，人脸框的偏移量[batch,4]，关键点的偏移量[batch,10]。

**对于是否存在人脸的分类：**损失只通过neg和pos数据来对参数进行更新，具体办法是通过label中的类别值做了一个遮罩来划分数据，只计算neg和pos的损失，不计算其他数据的损失；

**人脸框的回归：**损失只计算pos和part数据的；

**关键点的回归：**损失只计算landmark的。

论文中有个小技巧就是只通过前70%的数据进行更新参数，说是模型准确率会有提升，在代码中也都有体现，具体实现可以参考代码。

- 进入`triain_models/`目录
- 运行 `train_PNet.py`
读入`imglists/PNet/train_PNet_landmark.tfrecord_shuffle`文件开始训练Pnet。

## 4.3 RNet数据产生和训练
至此PNet训练结束，都不需要NMS来合并重合窗口，只有当生成下一个网络的输入数据时才用到。
RNet的landmark数据和PNet一样，是对带有关键点的图像截取得到，但要resize成24x24的作为输入。

RNet的pos,part,neg的数据是通过PNet得到的，图片还要以一定值来缩小尺度做成图像金字塔目的是获取更多可能的人脸框，人脸框中有人的概率大于一定阈值才保留，还要进行一定阈值的非极大值抑制，将太过重合的人脸框除掉，将PNet预测的人脸框于原图上截取，与真实人脸框的最大iou值来划分neg，pos,part数据，并resize成24作为RNet的输入。

RNet的损失函数和PNet相同，不同的是三种损失所占的比例不同， 可以参考我的博客：[MTCNN算法笔记](https://blog.csdn.net/weixin_41965898/article/details/84589666)损失函数那一节。

- 进入`triain_data/`目录

- 运行 `gen_hard_example.py -t PNet`

注意是` -t PNet`，生成三种RNet数据，写入目标`train_data/24/neg or pos or part`
观察PNet推荐框和真实数据情况如果 IOU < 0.3 负样本 0.4 - 0.65 part样本 0.65以上 正样本
- 运行 `gen_landmark_aug_24.py`

生成RNet的landmark数据，过程同gen_landmark_aug_12.py
- 运行 `gen_imglist_rnet.py`

整合 pos + neg + part + landmark，将`train_data/24`目录中`neg_24.txt、pos_24.txt、part_24.txt`和`landmark_24_aug.txt`整合到`train_data\imglists\PNet\train_RNet_landmark.txt`。
整合规则是全取。比例部分不在这里完成。
- 运行 `gen_RNet_tfrecords.py`
生成tfrecords文件，注意RNet是生成四个tfrecord到`train_data\imglists\RNet`，不是一个tfrecord。
landmark_landmark.tfrecord
pos_landmark.tfrecord
neg_landmark.tfrecord
part_landmark.tfrecord
- 进入`triain_models/`目录
- 运行 `train_RNet.py` 
训练RNet，按比例（1：3：1：1）的pos,neg,part,landmark从tfrecord中取样本。

## 4.4 ONet数据产生和训练
- 进入`triain_data/`目录
- 运行 `gen_hard_example.py -t RNet`

注意是` -t RNet`，过程同gen_hard_example.py -t PNet
- 运行 `gen_landmark_aug_48.py` 

过程同gen_landmark_aug_24.py
- 运行 `gen_imglist_onet.py`

过程同gen_imglist_rnet.py
- 运行 `gen_ONet_tfrecords.py`

过程同gen_RNet_tfrecords.py
- 进入`triain_models/`目录

- 运行 `train_ONet.py` 

过程同train_RNet.py

## 4.5 测试
- 进入`test/`目录

- 运行 `one_image_test.py`

# 参考
- https://github.com/AITTSMD/MTCNN-Tensorflow
- https://github.com/LeslieZhoa/tensorflow-MTCNN
