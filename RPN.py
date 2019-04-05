# -*- coding: utf-8 -*-
#输入应该有当前图片的所有groundtruth和label 和featuremap[32*32]*256
import tensorflow as tf
import numpy as np
from anchor import get_anchors,filter_anchor,MoveAndScala,return_anchor_label
import GLOBAL_CONSTANTS as GLOBAL

dx=np.arange(8,520,16)
dy=np.arange(8,520,16)
Coordinate_CentraOfAnchor=[]
Coordinate_Anchor=[]#生成1024*9长度的列表，每个anchor元素为[lx,ly,rx,ry]lx,ly为左上角坐标，rx,ry为右下角坐标


def weight_varible(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

for i in dx:
    for j in dy:
        Coordinate_CentraOfAnchor.append([j,i])
for i in Coordinate_CentraOfAnchor:
    #print(get_anchors(Coordinate_CentraOfAnchor[i]))
    AnchorTuple=list(get_anchors(i))
    Coordinate_Anchor.extend(AnchorTuple)
    #32*32*9
Coordinate_Anchor=tf.convert_to_tensor(Coordinate_Anchor)
W_conv3x3=weight_varible([3,3,256,256])#RPN首先对featuremap进行3x3卷积
b_conv3x3=bias_variable([256])

W_conv1x1_prob=weight_varible([1,1,256,18])#1*1卷积核 生成[32*32]*9*2
b_conv1x1_prob=bias_variable([18])

W_conv1x1_coordinate=weight_varible([1,1,256,36])#1*1卷积核 生成[32*32]*9*4
b_conv1x1_coordinate=bias_variable([36])
def from_index_get_anchorcoor():
    return Coordinate_Anchor
def smoothL1loss(x):
    return tf.cond(tf.abs(x)<1,lambda:0.5*x*x,lambda:tf.abs(x)-0.5)
    
def RPNOutput(featuremap,GL):#GL是[groundtruth label] n*5维数组 featuremap32*32*256
    h_conv3x3=tf.nn.relu(conv2d(featuremap,W_conv3x3)+b_conv3x3)  # 卷积层+relu层
    RPN_cls=conv2d(h_conv3x3,W_conv1x1_prob)+b_conv1x1_prob#32*32*18
    RPN_cls=tf.reshape(RPN_cls,[32,32,9,2])#生成softmax最后的二维
    RPN_cls=tf.cast(RPN_cls,tf.float64)
    RPN_reg=conv2d(h_conv3x3,W_conv1x1_coordinate)+b_conv1x1_coordinate#32*32*36
    RPN_reg=tf.reshape(RPN_reg,[32,32,9,4])
    RPN_reg=tf.cast(RPN_reg,tf.float64)
    #生成32*32*9个anchor
    LabelsOfAnchor=[]
    ListOfProposallabel=[]
    
    for i in range(GLOBAL.FLAGS.TotalNumOfAnchor):#32*32*9
        #print(GLOBAL.FLAGS.TotalNumOfAnchor)
        LabelsOfAnchor.append(filter_anchor(Coordinate_Anchor[i],GL))#tiaoshi
    
        
    LabelsOfAnchorTensor=tf.convert_to_tensor(LabelsOfAnchor)
    
    Label_1_location=tf.where(LabelsOfAnchorTensor>0)#返回前景标签的位置index shape=（？，1）的array tensor
    Label_0_location=tf.where(tf.equal(LabelsOfAnchorTensor,0))
        
    
    #NumOfPositive=tf.shape(Label_1_location)[0]#返回标签为前景的个数
    #NumOfNegative=tf.shape(Label_0_location)[0]#返回标签为背景的个数
    
    
    Random1=tf.random_shuffle(Label_1_location)
    Random0=tf.random_shuffle(Label_0_location)
    ListOfProposal1=Random1[:128]
    ListOfProposal2=Random0[:128]
    ListOfProposal=tf.concat([ListOfProposal1,ListOfProposal2],0)
    ListOfProposal=tf.cast(ListOfProposal,tf.int32)
    for H in range(128):
        ListOfProposallabel.append(return_anchor_label(Coordinate_Anchor[ListOfProposal1[H][0]],GL))
    for H in range(128):
        ListOfProposallabel.append(0)

    #下面计算rpn_softmax_loss 
    rpn_softmax_loss=0.0
    rpn_smoothl1_loss=0.0
    for i in range(128):
        Dim=ListOfProposal[i][0]
        dim0=Dim//(32*9)
        dim1=(Dim-32*9*dim0)//9
        dim2=Dim-32*9*dim0-9*dim1
        GLabel=LabelsOfAnchorTensor[Dim]-1#代表当前anchor预测的groundtruth
        
        TrueMoveAndScala=MoveAndScala(Coordinate_Anchor[Dim],GL[GLabel,:4])
        PreMoveAndScala=MoveAndScala(RPN_reg[dim0,dim1,dim2],GL[GLabel,:4])
        
        for num_4 in range(4):
            rpn_smoothl1_loss+=smoothL1loss(PreMoveAndScala[num_4]-TrueMoveAndScala[num_4])
        TensorPositive=RPN_cls[dim0,dim1,dim2]
        TensorPositive=tf.cast(TensorPositive,tf.float32)#注意softmax的传入必须要是浮点数
        SoftMax=tf.nn.softmax(TensorPositive)
        SoftPositiveLoss=-tf.log(tf.clip_by_value(SoftMax[0],1e-10,1.0))
        rpn_softmax_loss+=SoftPositiveLoss
        
    for i in range(128,256):#背景anchor不做smoothl1loss
        Dim=ListOfProposal[i][0]
        dim0=Dim//(32*9)
        dim1=(Dim-32*9*dim0)//9
        dim2=Dim-32*9*dim0-9*dim1
        TensorPositive=RPN_cls[dim0,dim1,dim2]
        TensorPositive=tf.cast(TensorPositive,tf.float32)#注意softmax的传入必须要是浮点数
        SoftMax=tf.nn.softmax(TensorPositive)
        SoftNegativeLoss=-tf.log(tf.clip_by_value(SoftMax[1],1e-10,1.0))
        rpn_softmax_loss+=SoftNegativeLoss
    
    rpn_softmax_loss=rpn_softmax_loss/256
    #rpn_softmax_loss=tf.cast(rpn_softmax_loss,tf.float64)
    rpn_smoothl1_loss=rpn_smoothl1_loss/256
    rpn_smoothl1_loss=tf.cast(rpn_smoothl1_loss,tf.float32)
    #rpn_softmax_loss，rpn_smoothl1_loss计算结束
    ListOfProposallabel=tf.convert_to_tensor(ListOfProposallabel)
    #ListOfProposallabel=tf.cast(ListOfProposallabel,tf.int64)
    return rpn_softmax_loss,rpn_smoothl1_loss,ListOfProposal,ListOfProposallabel

#此处不做nms实现 论文中采取重新生成32*32*9，再通过NMS和RPNloss的第一次迭代结果选择大概300个Bbox 
#这里直接将在RPN_loss随机选择的128个AOI和128个背景 或者当AOI小于128 由背景凑够256个传入到ROI pooling
#def fristamend(tensor,ListOfProposal)这里不做实现 进入到roi层不做回归
    
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        