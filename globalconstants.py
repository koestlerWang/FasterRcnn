# # -*- coding: utf-8 -*-
# import tensorflow as tf
#
# FLAGS=tf.app.flags.FLAGS
# tf.app.flags.DEFINE_integer('max_width',512,"the width of the input picture")
# tf.app.flags.DEFINE_integer('max_height',512,"the height of the input picture")
# tf.app.flags.DEFINE_integer('NumOfAnchor',9,"each district produces nine anchors")
# tf.app.flags.DEFINE_integer('batch_size',1,"trainning batch size")#注意faster rcnn每次只拿一张图片进行训练
# tf.app.flags.DEFINE_integer('NumOfPicture',10,"total number of pictures")#只做fasterrcnn的实例demo，利用下载的十张照片作为训练集
# tf.app.flags.DEFINE_integer('NumOfLabels',3,"the head,the foot and the tail")
# tf.app.flags.DEFINE_integer('TotalNumOfAnchor',32*32*9,"total number of anchor each picture")
#
# #由于训练的图片比较少，在这里直接将boundingbox位置和标签写在内存里
# tf.app.flags.DEFINE_integer('Cat1Head',[46,55,188,242],"only one")#第一张训练图片
# tf.app.flags.DEFINE_integer('Cat1Foot1',[158,331,208,450],"many")
# tf.app.flags.DEFINE_integer('Cat1Foot2',[206,293,258,450],"many")
# tf.app.flags.DEFINE_integer('Cat1Foot3',[289,402,385,451],"many")
# tf.app.flags.DEFINE_integer('Cat1Tail',[392,193,489,366],"only one")
# tf.app.flags.DEFINE_integer('Cat1Lables',[1,2,2,2,3,-1],"labels")
# tf.app.flags.DEFINE_integer('picture1',[[46,55,188,242,1],[158,331,208,450,2],[206,293,258,450,2],[289,402,385,451,2],[392,193,489,366,3],[0,0,0,0,-1]],"")
#
# tf.app.flags.DEFINE_integer('Cat2Head',[131,71,268,212],"only one")#第二张训练图片
# tf.app.flags.DEFINE_integer('Cat2Foot1',[94,290,133,390],"many")
# tf.app.flags.DEFINE_integer('Cat2Foot2',[101,339,209,444],"many")
# tf.app.flags.DEFINE_integer('Cat2Foot3',[248,390,304,437],"many")
# tf.app.flags.DEFINE_integer('Cat2Tail',[305,240,431,343],"only one")
# tf.app.flags.DEFINE_integer('Cat2Lables',[1,2,2,2,3,-1],"labels")
# tf.app.flags.DEFINE_integer('picture2',[[131,71,268,212,1],[94,290,133,390,2],[101,339,209,444,2],[248,390,304,437,2],[305,240,431,343,3],[0,0,0,0,-1]],"")
#
# tf.app.flags.DEFINE_integer('Cat3Head',[125,40,237,206],"only one")#第三张训练图片
# tf.app.flags.DEFINE_integer('Cat3Foot1',[150,349,184,481],"many")
# tf.app.flags.DEFINE_integer('Cat3Foot2',[179,367,238,489],"many")
# tf.app.flags.DEFINE_integer('Cat3Foot3',[223,444,281,476],"many")
# tf.app.flags.DEFINE_integer('Cat3Tail',[302,385,478,455],"only one")
# tf.app.flags.DEFINE_integer('Cat3Lables',[1,2,2,2,3,-1],"labels")
# tf.app.flags.DEFINE_integer('picture3',[[125,40,237,206,1],[150,349,184,481,2],[179,367,238,489,2],[223,444,281,476,2],[302,385,478,455,3],[0,0,0,0,-1]],"")
#
# tf.app.flags.DEFINE_integer('Cat4Head',[135,189,240,329],"only one")#第四张训练图片
# tf.app.flags.DEFINE_integer('Cat4Foot1',[80,374,162,434],"many")
# tf.app.flags.DEFINE_integer('Cat4Foot2',[155,393,252,440],"many")
# tf.app.flags.DEFINE_integer('Cat4Foot3',[352,357,384,398],"many")
# tf.app.flags.DEFINE_integer('Cat4Foot4',[374,339,414,411],"many")
# tf.app.flags.DEFINE_integer('Cat4Tail',[357,275,418,332],"only one")
# tf.app.flags.DEFINE_integer('Cat4Lables',[1,2,2,2,2,3],"labels")
# tf.app.flags.DEFINE_integer('picture4',[[135,189,240,329,1],[80,374,162,434,2],[155,393,252,440,2],[352,357,384,398,2],[374,339,414,411,2],[357,275,418,332,3]],"")
#
# tf.app.flags.DEFINE_integer('Cat5Head',[93,68,230,251],"only one")#第五张训练图片
# tf.app.flags.DEFINE_integer('Cat5Foot1',[97,311,168,452],"many")
# tf.app.flags.DEFINE_integer('Cat5Foot2',[153,316,237,435],"many")
# tf.app.flags.DEFINE_integer('Cat5Foot3',[237,351,319,399],"many")
# tf.app.flags.DEFINE_integer('Cat5Foot4',[357,331,413,416],"only one")
# tf.app.flags.DEFINE_integer('Cat5Tail',[371,214,457,376],"only one")
# tf.app.flags.DEFINE_integer('Cat5Lables',[1,2,2,2,2,3],"labels")
# tf.app.flags.DEFINE_integer('picture5',[[93,68,230,251,1],[97,311,168,452,2],[153,316,237,435,2],[237,351,319,399,2],[357,331,413,416,2],[371,214,457,376,3]],"")
#
# tf.app.flags.DEFINE_integer('Cat6Head',[237,24,400,168],"only one")#第六张训练图片
# tf.app.flags.DEFINE_integer('Cat6Foot1',[195,207,288,389],"many")
# tf.app.flags.DEFINE_integer('Cat6Foot2',[254,376,312,422],"many")
# tf.app.flags.DEFINE_integer('Cat6Foot3',[326,216,405,478],"many")
# tf.app.flags.DEFINE_integer('Cat6Tail',[26,170,183,332],"only one")
# tf.app.flags.DEFINE_integer('Cat6Lables',[1,2,2,2,3,-1],"labels")
# tf.app.flags.DEFINE_integer('picture6',[[237,24,400,168,1],[195,207,288,389,2],[254,376,312,422,2],[326,216,405,478,2],[26,170,183,332,3],[0,0,0,0,-1]],"")
#
# tf.app.flags.DEFINE_integer('Cat7Head',[337,48,489,226],"only one")#第七张训练图片
# tf.app.flags.DEFINE_integer('Cat7Foot1',[114,312,185,417],"many")
# tf.app.flags.DEFINE_integer('Cat7Foot2',[175,306,293,425],"many")
# tf.app.flags.DEFINE_integer('Cat7Foot3',[301,273,386,436],"many")
# tf.app.flags.DEFINE_integer('Cat7Tail',[5,214,111,326],"only one")
# tf.app.flags.DEFINE_integer('Cat7Lables',[1,2,2,2,3,-1],"labels")
# tf.app.flags.DEFINE_integer('picture7',[[337,48,489,226,1],[114,312,185,417,2],[175,306,293,425,2],[301,273,386,436,2],[5,214,111,326,3],[0,0,0,0,-1]],"")
#
# tf.app.flags.DEFINE_integer('Cat8Head',[256,120,406,307],"only one")#第八张训练图片
# tf.app.flags.DEFINE_integer('Cat8Foot1',[184,371,235,448],"many")
# tf.app.flags.DEFINE_integer('Cat8Foot2',[280,359,373,483],"many")
# tf.app.flags.DEFINE_integer('Cat8Foot3',[357,343,440,457],"many")
# tf.app.flags.DEFINE_integer('Cat8Tail',[195,51,278,216],"only one")
# tf.app.flags.DEFINE_integer('Cat8Lables',[1,2,2,2,3,-1],"labels")
# tf.app.flags.DEFINE_integer('picture8',[[256,120,406,307,1],[184,371,235,448,2],[280,359,373,483,2],[357,343,440,457,2],[195,51,278,216,3],[0,0,0,0,-1]],"")
#
# tf.app.flags.DEFINE_integer('Cat9Head',[61,141,218,313],"only one")#第九张训练图片
# tf.app.flags.DEFINE_integer('Cat9Foot1',[74,385,127,444],"many")
# tf.app.flags.DEFINE_integer('Cat9Foot2',[114,376,239,446],"many")
# tf.app.flags.DEFINE_integer('Cat9Tail',[257,339,458,433],"only one")
# tf.app.flags.DEFINE_integer('Cat9Lables',[1,2,2,3,-1,-1],"labels")
# tf.app.flags.DEFINE_integer('picture9',[[61,141,218,313,1],[74,385,127,444,2],[114,376,239,446,2],[257,339,458,433,3],[0,0,0,0,-1],[0,0,0,0,-1]],"")
#
# tf.app.flags.DEFINE_integer('Cat10Head',[301,83,406,213],"only one")#第十张训练图片
# tf.app.flags.DEFINE_integer('Cat10Foot1',[220,373,302,471],"many")
# tf.app.flags.DEFINE_integer('Cat10Foot2',[318,303,381,476],"many")
# tf.app.flags.DEFINE_integer('Cat10Foot3',[368,312,425,409],"many")
# tf.app.flags.DEFINE_integer('Cat10Tail',[57,332,193,453],"only one")
# tf.app.flags.DEFINE_integer('Cat10Lables',[1,2,2,2,3,-1],"labels")
# tf.app.flags.DEFINE_integer('picture10',[[301,83,406,213,1],[220,373,302,471,2],[318,303,381,476,2],[368,312,425,409,2],[57,332,193,453,3],[0,0,0,0,-1]],"")
#
# #tf.app.flags.DEFINE_float=('AnchorArea',[200*200,100*100,50*50],"the three size of anchor area")
# #tf.app.flags.DEFINE_flaot=('AnchorScala',[1/1,1/2,2/1],"height-width ratio")此处可以直接算出高度和宽度 如下
# tf.app.flags.DEFINE_float('WidthAndHeight',[[50.,50.],[35.,70.],[70.,35.],[100.,100.],[70.,140.],[140.,70.],[200.,200.],[140.,280.],[280.,140.]],"the nine sizes of boundingbox")
#
#
#
# tf.app.flags.DEFINE_string('TrainPicturesPath','C://Users//wgh//Desktop//cat_trainning//',"path of trainning pictures")
# tf.app.flags.DEFINE_string('TestPicturesPath','C://Users//wgh//Desktop//cat_testing//',"path of testing pictures")
#
