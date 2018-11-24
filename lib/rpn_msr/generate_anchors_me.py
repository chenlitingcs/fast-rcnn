import numpy as np
# 功能描述：生成多尺度、多宽高比的anchors.
#          尺度为：128,256,512; 宽高比为：1:2,1:1,2:1
# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#最终结果生成导向
#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=16, ratios=[0.5, 1, 2], #param ratios:表示需要产生的anchors的纵横比,通常是含有多个纵横比数值的list.
                     scales=2**np.arange(3, 6)):
    base_anchor = np.array([1,1,base_size,base_size])-1
    ratio_anchors = _ratio_enum(base_anchor,ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i,:],scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):#因为函数实际情况,形参主动和实参一致也是常见情况,真的很常见,自己也要主动用起来.


    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w,h,x_ctr,y_ctr

def _mkanchors(ws,hs,x_ctr,y_ctr):
    ws = ws[:,np.newaxis]
    hs =hs[:,np.newaxis]  #ws,hs后面不是括号，而且也不是把= 后的全写在中括号内，原因是：
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(base_anchor,ratios):
    w, h, x_ctr, y_ctr = _whctrs(base_anchor)
    size = w*h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(size/ws)
    anchors = _mkanchors(ws,hs,x_ctr,y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr =  _whctrs(anchor)        #每个子函数一个这行，这样做有什么原理支撑
    ws = w * scales
    hs = h * scales
    anchors =_mkanchors(ws,hs,x_ctr,y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed; embed() #在程序的最后加上这一整句代码,可以在run窗口输入变量名,进而查看变量
'''

#base_size =16 是因为从conv4层出来的最小的检测大小是16个像素
#在 fast_rcnn/config.py中有定义：__C.TRAIN.RPN_MIN_SIZE = 16
#生成anchors总函数：ratios为一个列表,表示宽高比为1:2,1:1,2:1
#2**x表示:2^x,scales:[2^3 2^4 2^5],即:[8 16 32]
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):#scales还不如直接写[8,16.32]
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window. 新建一个数组：base_anchor:[0 0 15 15]     #～对应上面的anchors转向array,每行表示一个anchor
    枚举一个anchor的各种尺度,以anchor[0 0 15 15]为例,scales[8 16 32]

    """
    base_anchor =np.array([1, 1, base_size, base_size]) - 1  #新建一个数组：base_anchor:[0 0 15 15]，到底为什么-1  一个base_anchor坐标为[0，0，15，15]，因为电脑是从0开始计数的（左上角嘛），其实是[1，1，16，16]，后面计算坐标时候也要考虑到
    ratio_anchors = _ratio_enum(base_anchor, ratios) #枚举举各种宽高比，生成三个比例的anchor
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)  #枚举各种尺度,vstack:竖向合并数组
    for i in range(ratio_anchors.shape[0])]) #shape[0]:读取矩阵第一维长度,其值为3(涉及到上面某个不需要看实现的函数）
    return anchors


def _whctrs(anchor):#

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)  #anchor中心点坐标
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

#给定一组宽高向量,输出各个anchor的左上角和右下角的坐标,即预测窗口,**输出anchor的面积相等,只是宽高比不同**
def _mkanchors(ws, hs, x_ctr, y_ctr):
    #ws:[23 16 11],hs:[12 16 22],ws和hs一一对应.
    # 将ws和hs增加一个维度,使其变成一个二维数组 ws = ws[:, np.newaxis] hs = hs[:, np.newaxis]
    ws = ws[:, np.newaxis]  #newaxis:为数组增加一个轴, :在前还要转置
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),    #hstack、vstack:合并数组,计算所有的anchors的坐标（左上角和右下角）
                         y_ctr - 0.5 * (hs - 1),    #anchors：[[-3.5 2 18.5 13]
                         x_ctr + 0.5 * (ws - 1),     #         [0  0  15  15]  #!!!!
                         y_ctr + 0.5 * (hs - 1)))    #        [2.5 -3 12.5 18]]                    提出一个问题：这样就可以算anchor的雏形了，这个怎么导出来成为左上角和右下角的坐标的
    return anchors

#枚举一个anchor的各种宽高比,anchor[0 0 15 15],ratios[0.5,1,2]
def _ratio_enum(anchor, ratios):
    """  函数功能： 生成1：2  1:1 2:1的锚点
    Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    #返回宽高和中心坐标,w:16,h:16,x_ctr:7.5,y_ctr:7.5 w, h, x_ctr, y_ctr = _whctrs(anchor)
     #计算一个基础size,w*h=256 size = w * h #得到比例的size_ratios,type：np.array #为（512,256,128）
      size_ratios = size / ratios #np.sqrt开方 #ws:[23 16 11]  与  hs:[12 16 22] ws = np.round(np.sqrt(size_ratios))

    w, h, x_ctr, y_ctr = _whctrs(anchor)  #返回宽高和中心坐标,w:16,h:16,x_ctr:7.5,y_ctr:7.5
    size = w * h   #size:16*16=256
    size_ratios = size / ratios  #256/ratios[0.5,1,2]=[512,256,128]
    #round()方法返回x的四舍五入的数字,sqrt()方法返回数字x的平方根
    ws = np.round(np.sqrt(size_ratios)) #ws:[23 16 11]
    hs = np.round(ws * ratios)    #hs:[12 16 22],ws和hs一一对应.比如23&12
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

#枚举一个anchor的各种尺度,以anchor[0 0 15 15]为例,scales[8 16 32]
def _scale_enum(anchor, scales):
    """   列举关于一个anchor的三种尺度 128*128,256*256,512*512
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor) #返回宽高和中心坐标,w:16,h:16,x_ctr:7.5,y_ctr:7.5,因为在之前返回anchors：[[-3.5 2 18.5 13]
                                                                                                             #[0  0  15  15]
                                                                                                             #[2.5 -3 12.5 18]]所以产生的9个框的位置也不一样,操,这么复杂
    ws = w * scales   #[128 256 512]
    hs = h * scales   #[128 256 512]
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr) #[[-55 -55 72 72] [-119 -119 136 136] [-247 -247 264 264]]
    return anchors
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)): """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.

 
    """ # 产生基准的anchor,所有的anchors都是将这个anchor变形缩放后得到的.
    base_anchor = np.array([1, 1, base_size, base_size]) - 1 # 对基准的anchor进行纵横比的变形,在变形的过程中尽可能保持anchor的总面积不变.
# ratio_anchors = _ratio_enum(base_anchor, ratios) # 对经过纵横比变化之后的anchors进行尺寸上的缩放.
# anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in xrange(ratio_anchors.shape[0])]) # 返回所有生成的anchors.
#  return anchors # 根据给定的anchor的信息,返回其对应的宽度,高度,以及中心的坐标x,y的值.
def _whctrs(anchor): """
    Return width, height, x center, and y center for an anchor (window).
    :param anchor: 通常是一个长度为4的list或者numpy数组,表示需要获取宽度,高度和中心坐标的anchor.
    :return: 该函数返回四个数值,分别表示anchor宽度 w,高度 h,中心的x坐标 x_ctr, 中心的y坐标 y_ctr.
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr # 与上面的函数相反,该函数是给定anchor(s)的宽,高,中心点的x和y值,生成对应的anchor(s).'''


