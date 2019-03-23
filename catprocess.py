# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:28:06 2019

@author: wgh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:16:37 2019

@author: wgh
"""

import os
import tensorflow as tf


path="C://Users//wgh//Desktop//cat"
classes=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']

with tf.Session() as sess:
    class_path=path+'//'
    for imagename in os.listdir(class_path):
        imag_path=class_path+imagename
            
        img_raw=tf.gfile.GFile(imag_path,'rb').read()
             #print(type(img_raw))
        img=tf.image.decode_jpeg(img_raw)
            #print(type(img))
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        resized=tf.image.resize_images(img,[500,500],method=0)
            #print(type(resized))
            # 转换图像的数据类型
        img_data = tf.image.convert_image_dtype(resized, dtype=tf.uint8)
        encoded_image=tf.image.encode_jpeg(img_data)
        with tf.gfile.GFile(imag_path,"wb") as f:
            f.write(encoded_image.eval())
                
