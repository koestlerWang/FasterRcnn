# -*- coding: utf-8 -*-

import tensorflow as tf
from RPN import RPNOutput,from_index_get_anchorcoor
import GLOBAL_CONSTANTS as GLOBAL
#from tensorflow.python import debug as tf_debug
import os
GL_list=[]
GL_list.append(GLOBAL.FLAGS.picture1.copy())
GL_list.append(GLOBAL.FLAGS.picture2.copy())
GL_list.append(GLOBAL.FLAGS.picture3.copy())
GL_list.append(GLOBAL.FLAGS.picture4.copy())
GL_list.append(GLOBAL.FLAGS.picture5.copy())
GL_list.append(GLOBAL.FLAGS.picture6.copy())
GL_list.append(GLOBAL.FLAGS.picture7.copy())
GL_list.append(GLOBAL.FLAGS.picture8.copy())
GL_list.append(GLOBAL.FLAGS.picture9.copy())
GL_list.append(GLOBAL.FLAGS.picture10.copy())


def weight_varible(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get2x2box(anchor):#ROI pooling
    lx=anchor[0]
    ly=anchor[1]
    rx=anchor[2]
    ry=anchor[3]
    mx=(lx+rx)//2
    my=(ly+ry)//2
    return([lx,ly,mx,my],[mx,ly,rx,my],[lx,my,mx,ry],[mx,my,rx,ry])
    


#存放十张图片的groundtruth和label
# 定义输入的placeholder
x=tf.placeholder(tf.float32,shape=[1,512,512,3])#特征
GL=tf.placeholder(tf.int32,shape=[6,5])
#第一个卷积层
W_conv1=weight_varible([7,7,3,32])#代表卷积核尺寸为7*7 RGB channel64
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x,W_conv1)+b_conv1)  # 卷积层+relu层
h_pool1=max_pool_2x2(h_conv1)#缩小尺寸
#output [256*256]*32

# 第二个卷积层
W_conv2=weight_varible([7,7,32,64])  
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
#output [128*128]*64

# 第三个卷积层
W_conv3=weight_varible([7,7,64,128])  
b_conv3=bias_variable([128])
h_conv3=tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
h_pool3=max_pool_2x2(h_conv3)
#output [64*64]*128

# 第四个卷积层
W_conv4=weight_varible([7,7,128,256])  
b_conv4=bias_variable([256])
h_conv4=tf.nn.relu(conv2d(h_pool3,W_conv4)+b_conv4)
h_pool4=max_pool_2x2(h_conv4)

#output 1*[32*32]*256
#RPN
'''
ListOfProposal=[[22],[44],[61]]
ListOfProposal=tf.convert_to_tensor(ListOfProposal)
ListOfProposal=tf.cast(ListOfProposal,tf.int64)'''
rpn_softmax_loss,rpn_smoothl1_loss,ListOfProposal,ListOfProposallabel=RPNOutput(h_pool4,GL)#ListOfProposal,ListOfProposallabel为nparray
#(<tf.Tensor dtype=float64>, <tf.Tensor dtype=float64>, <tf.Tensor ) dtype=int64>, <tf.Tensor ) dtype=int64>)
#定义映射关系
Box_train=[]
Get_Anchor=from_index_get_anchorcoor()
for i in range(256):
    ANCHOR=Get_Anchor[ListOfProposal[i][0]]//16#[lx,ly,rx,ry]list
    box1,box2,box3,box4=get2x2box(ANCHOR)
    box1tensor=tf.reduce_max(h_pool4[box1[1]:box1[3]+1,box1[0]:box1[2]+1],[0,1,2])
    box2tensor=tf.reduce_max(h_pool4[box2[1]:box2[3]+1,box2[0]:box2[2]+1],[0,1,2])
    box3tensor=tf.reduce_max(h_pool4[box3[1]:box3[3]+1,box3[0]:box3[2]+1],[0,1,2])
    box4tensor=tf.reduce_max(h_pool4[box4[1]:box4[3]+1,box4[0]:box4[2]+1],[0,1,2])
   
    box_all_tensor=tf.stack([box1tensor,box2tensor,box3tensor,box4tensor],0)
    BoxTensor=tf.reshape(box_all_tensor,[2,2,256])
    Box_train.append(BoxTensor)
Box_train=tf.stack(Box_train,0)
# 定义一个全连接层
W_fc1=weight_varible([2*2*256,1024])  # 全连接层隐含节点为1024个
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(Box_train,[256,2*2*256])  # 对第二个卷积层的输出tensor进行变形，将其转化为1D的向量
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
# 为了减轻过拟合，下面使用一个dropout层
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
# 将dropout层的输出连接一个softmax层，得到最后的概率输出
W_fc2=weight_varible([1024,4])
b_fc2=bias_variable([4])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)#[256*4]
ListOfProposallabel_OneHot=tf.one_hot(ListOfProposallabel,4)#[256*4]
# 定义损失函数为cross entronpy，优化器使用Adam
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ListOfProposallabel_OneHot*tf.log(y_conv),reduction_indices=[1]))

#到此已经得到三种loss
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy+rpn_softmax_loss+rpn_smoothl1_loss)  # 学习率为1e-4
#accuracy=
# 定义评测准确率的操作
def main(Useless):
    print('start')      
    saver = tf.train.Saver()
 
#判断模型保存路径是否存在，不存在就创建
    if not os.path.exists('tmp/'):
        os.mkdir('tmp/')

    sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    for i in range(1):  # 进行训练
        for j in range(1):
            img_raw=tf.gfile.GFile(GLOBAL.FLAGS.TrainPicturesPath+'cat'+str(i+1)+'.jpg','rb').read()
            img=tf.image.decode_jpeg(img_raw)
            img=tf.image.convert_image_dtype(img,dtype=tf.float32)
            img=tf.reshape(img, [1,512,512,3])
            sess.run(train_step,feed_dict={x:sess.run(img),GL:GL_list[i],keep_prob:0.5})
        saver.save(sess,'tmp/FasterRcnn.ckpt',global_step=i)
# 全部训练完成后，在最终的测试集上进行全面的测试，得到整体的分类准确率
if __name__=='__main__':
    tf.app.run()
