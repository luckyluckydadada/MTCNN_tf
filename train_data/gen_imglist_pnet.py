import numpy as np
import numpy.random as npr
import os

data_dir = '.'
#anno_file = os.path.join(data_dir, "anno.txt")

'''将pos,part,neg,landmark四者混在一起'''
size = 12

if size == 12:
    net = "PNet"
elif size == 24:
    net = "RNet"
elif size == 48:
    net = "ONet"
# 读取人脸位置 负样本位置为0 正样本为xywh
with open(os.path.join(data_dir, '%s/pos_%s.txt' % (size, size)), 'r') as f:
    pos = f.readlines()

with open(os.path.join(data_dir, '%s/neg_%s.txt' % (size, size)), 'r') as f:
    neg = f.readlines()

with open(os.path.join(data_dir, '%s/part_%s.txt' % (size, size)), 'r') as f:
    part = f.readlines()
# 读取人脸关键点
with open(os.path.join(data_dir,'%s/landmark_%s_aug.txt' %(size,size)), 'r') as f:
    landmark = f.readlines()
    
dir_path = os.path.join(data_dir, 'imglists')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
if not os.path.exists(os.path.join(dir_path, "%s" %(net))):
    os.makedirs(os.path.join(dir_path, "%s" %(net)))
with open(os.path.join(dir_path, "%s" %(net),"train_%s_landmark.txt" % (net)), "w") as f:
    nums = [len(neg), len(pos), len(part)]
    ratio = [3, 1, 1]
    #base_num = min(nums)
    base_num = 250000
    print('neg数量：{} pos数量：{} part数量:{} 基数:{}'.format(len(neg),len(pos),len(part),base_num))
    # 选择需要保留的负、正、部分样本个数
    if len(neg) > base_num * 3: #neg数量大于3倍base
        neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True) #从所有neg中取3*base个样本
    else:
        neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
    pos_keep = npr.choice(len(pos), size=base_num, replace=True)
    part_keep = npr.choice(len(part), size=base_num, replace=True)
    print('neg数量：{} pos数量：{} part数量:{}'.format(len(neg_keep),len(pos_keep),len(part_keep)))

    for i in pos_keep:
        f.write(pos[i])
    for i in neg_keep:
        f.write(neg[i])
    for i in part_keep:
        f.write(part[i])
    for item in landmark:
        f.write(item)
# 输出到 imglists/PNet/train_PNet_landmark.txt 有 pos + neg + part + landmark 组成