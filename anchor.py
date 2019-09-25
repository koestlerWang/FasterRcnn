# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:59:46 2019

@author: wgh
"""
import tensorflow as tf
import globalconstants


def get_centra(coordinate):#coordinate is the input of the coordination(four numbers input)
    ListCoor=[]
    dx=tf.round((coordinate[0]+coordinate[2])/2)
    ListCoor.append(dx)
    dy=tf.round((coordinate[1]+coordinate[3])/2)
    ListCoor.append(dy)
    return tf.convert_to_tensor(ListCoor)


def get_anchors(coordinate):#coordinate is the input of the cnetra of the area(where feature-map reflect)  [width_coor,height_coor]
    
    dx=coordinate[0]
    dy=coordinate[1]
    
    Anchor0L_dx=int(dx-globalconstants.FLAGS.WidthAndHeight[0][0]/2)
    Anchor0L_dy=int(dy-globalconstants.FLAGS.WidthAndHeight[0][1]/2)
    Anchor0R_dx=int(dx+globalconstants.FLAGS.WidthAndHeight[0][0]/2)
    Anchor0R_dy=int(dy+globalconstants.FLAGS.WidthAndHeight[0][1]/2)
    Anchor0=[Anchor0L_dx,Anchor0L_dy,Anchor0R_dx,Anchor0R_dy]#produce anchor1
    
    Anchor1L_dx=int(dx-globalconstants.FLAGS.WidthAndHeight[1][0]/2)
    Anchor1L_dy=int(dy-globalconstants.FLAGS.WidthAndHeight[1][1]/2)
    Anchor1R_dx=int(dx+globalconstants.FLAGS.WidthAndHeight[1][0]/2)
    Anchor1R_dy=int(dy+globalconstants.FLAGS.WidthAndHeight[1][1]/2)
    Anchor1=[Anchor1L_dx,Anchor1L_dy,Anchor1R_dx,Anchor1R_dy]#produce anchor2
    
    Anchor2L_dx=int(dx-globalconstants.FLAGS.WidthAndHeight[2][0]/2)
    Anchor2L_dy=int(dy-globalconstants.FLAGS.WidthAndHeight[2][1]/2)
    Anchor2R_dx=int(dx+globalconstants.FLAGS.WidthAndHeight[2][0]/2)
    Anchor2R_dy=int(dy+globalconstants.FLAGS.WidthAndHeight[2][1]/2)
    Anchor2=[Anchor2L_dx,Anchor2L_dy,Anchor2R_dx,Anchor2R_dy]#produce anchor3
    
    Anchor3L_dx=int(dx-globalconstants.FLAGS.WidthAndHeight[3][0]/2)
    Anchor3L_dy=int(dy-globalconstants.FLAGS.WidthAndHeight[3][1]/2)
    Anchor3R_dx=int(dx+globalconstants.FLAGS.WidthAndHeight[3][0]/2)
    Anchor3R_dy=int(dy+globalconstants.FLAGS.WidthAndHeight[3][1]/2)
    Anchor3=[Anchor3L_dx,Anchor3L_dy,Anchor3R_dx,Anchor3R_dy]#produce anchor4
    
    Anchor4L_dx=int(dx-globalconstants.FLAGS.WidthAndHeight[4][0]/2)
    Anchor4L_dy=int(dy-globalconstants.FLAGS.WidthAndHeight[4][1]/2)
    Anchor4R_dx=int(dx+globalconstants.FLAGS.WidthAndHeight[4][0]/2)
    Anchor4R_dy=int(dy+globalconstants.FLAGS.WidthAndHeight[4][1]/2)
    Anchor4=[Anchor4L_dx,Anchor4L_dy,Anchor4R_dx,Anchor4R_dy]#produce anchor5
    
    Anchor5L_dx=int(dx-globalconstants.FLAGS.WidthAndHeight[5][0]/2)
    Anchor5L_dy=int(dy-globalconstants.FLAGS.WidthAndHeight[5][1]/2)
    Anchor5R_dx=int(dx+globalconstants.FLAGS.WidthAndHeight[5][0]/2)
    Anchor5R_dy=int(dy+globalconstants.FLAGS.WidthAndHeight[5][1]/2)
    Anchor5=[Anchor5L_dx,Anchor5L_dy,Anchor5R_dx,Anchor5R_dy]#produce anchor6
    
    Anchor6L_dx=int(dx-globalconstants.FLAGS.WidthAndHeight[6][0]/2)
    Anchor6L_dy=int(dy-globalconstants.FLAGS.WidthAndHeight[6][1]/2)
    Anchor6R_dx=int(dx+globalconstants.FLAGS.WidthAndHeight[6][0]/2)
    Anchor6R_dy=int(dy+globalconstants.FLAGS.WidthAndHeight[6][1]/2)
    Anchor6=[Anchor6L_dx,Anchor6L_dy,Anchor6R_dx,Anchor6R_dy]#produce anchor7
    
    Anchor7L_dx=int(dx-globalconstants.FLAGS.WidthAndHeight[7][0]/2)
    Anchor7L_dy=int(dy-globalconstants.FLAGS.WidthAndHeight[7][1]/2)
    Anchor7R_dx=int(dx+globalconstants.FLAGS.WidthAndHeight[7][0]/2)
    Anchor7R_dy=int(dy+globalconstants.FLAGS.WidthAndHeight[7][1]/2)
    Anchor7=[Anchor7L_dx,Anchor7L_dy,Anchor7R_dx,Anchor7R_dy]#produce anchor8
    
    Anchor8L_dx=int(dx-globalconstants.FLAGS.WidthAndHeight[8][0]/2)
    Anchor8L_dy=int(dy-globalconstants.FLAGS.WidthAndHeight[8][1]/2)
    Anchor8R_dx=int(dx+globalconstants.FLAGS.WidthAndHeight[8][0]/2)
    Anchor8R_dy=int(dy+globalconstants.FLAGS.WidthAndHeight[8][1]/2)
    Anchor8=[Anchor8L_dx,Anchor8L_dy,Anchor8R_dx,Anchor8R_dy]#produce anchor9
    
    return (Anchor0,Anchor1,Anchor2,Anchor3,Anchor4,Anchor5,Anchor6,Anchor7,Anchor8)#python中一般返回多个值都会使用元组，return的括号可以省略

def IOU(anchor,groundtruths):#计算每个anchor与所有groundtruth的IOU
    ListOfTensor=[]
    Area_anchor=(anchor[2]-anchor[0])*(anchor[3]-anchor[1])
    for i in range(6):
        Area_groundtruths=(groundtruths[i][2]-groundtruths[i][0])*(groundtruths[i][3]-groundtruths[i][1])
    
        x1=tf.maximum(anchor[0],groundtruths[i][0])
        x2=tf.minimum(anchor[2],groundtruths[i][2])
    
        y1=tf.maximum(anchor[1],groundtruths[i][1])
        y2=tf.minimum(anchor[3],groundtruths[i][3])
    
        width=tf.maximum(0,x2-x1)
        height=tf.maximum(0,y2-y1)
    
        Area_overlap=width*height
    
        IOU=Area_overlap/(Area_anchor+Area_groundtruths-Area_overlap)
        ListOfTensor.append(IOU)
        
    return tf.convert_to_tensor(ListOfTensor)
        

def filter_anchor(anchor,image_groundtruths):#使用过滤器过滤anchor n行5列
    ## 不越界但是IOU低于0.3的标记为0（背景） 不越界且IOU大于0.6标记为1（前景）
    IOU_tensor=IOU(anchor,image_groundtruths[:,:4])
    def True_process():
        #函数嵌套
        def Back_pre():
            return tf.cond(tf.greater(IOU_tensor[tf.argmax(IOU_tensor)],0.3),lambda:-1,lambda:0)
        try:
            return tf.cond(tf.greater(IOU_tensor[tf.argmax(IOU_tensor)],0.6),lambda:tf.cast(tf.argmax(IOU_tensor),dtype=tf.int32)+1,Back_pre)
                #返回第几个groundtruth,暂时不需要返回属于哪一类物体
        except Exception as e:
            return -1   
    def False_process():
        return -1
    def Tru():
        return 1
    def Fal():
        return 0
    Condition1=tf.cond(tf.greater(anchor[0],0),Tru,Fal)
    Condition2=tf.cond(tf.greater(anchor[1],0),Tru,Fal)
    Condition3=tf.cond(tf.greater(anchor[2],500),Fal,Tru)
    Condition4=tf.cond(tf.greater(anchor[3],500),Fal,Tru)
    process=tf.cond(tf.cast(Condition1*Condition2*Condition3*Condition4,dtype=tf.bool),True_process,False_process)
    return process
   
 ##越界的标记为-1  

def return_anchor_label(anchor,image_groundtruths):#这里的anchor已经被删选出来属于前景
    IOU_tensor=IOU(anchor,image_groundtruths[:,:4])
    return image_groundtruths[tf.argmax(IOU_tensor)][4]
              
               
def MoveAndScala(anchor,groundtruth):
    AXY=get_centra(anchor)
    GXY=get_centra(groundtruth)
    ax=AXY[0]
    ay=AXY[1]
    gx=GXY[0]
    gy=GXY[1]  
    WidthOfAnchor=anchor[2]-anchor[0]
    HeightOfAnchor=anchor[3]-anchor[1]
    
    WidthOfGroundtruth=groundtruth[2]-groundtruth[0]
    HeightOfGroundtruth=groundtruth[3]-groundtruth[1]
    
    WidthOfAnchor=tf.cast(WidthOfAnchor,tf.float64)
    HeightOfAnchor=tf.cast(HeightOfAnchor,tf.float64)
    WidthOfGroundtruth=tf.cast(WidthOfGroundtruth,tf.float64)
    HeightOfGroundtruth=tf.cast(HeightOfGroundtruth,tf.float64)
    
    dx=(gx-ax)/512
    dy=(gy-ay)/512#尽量归一化处理

    ScalaWidth=tf.log(WidthOfGroundtruth/WidthOfAnchor)
    ScalaHeight=tf.log(HeightOfGroundtruth/HeightOfAnchor)
    
    return (dx,dy,ScalaWidth,ScalaHeight)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
