import numpy as np
import yaml
from fast_rcnn.config import cfg ##配置文件
from generate_anchors import generate_anchors ##生成初始框
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
import pdb


DEBUG = False
"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors"). 该文件中的函数的主要目的是通过将估计的边界框（估计的？第一波修正的anchor，对于上帝视角的真图，当然是估计的）变换应用于一组常规框（称为“anchors”）来输出目标检测proposals。
说人话就是用bbox变换(函数应该是的,他就是第一次修正---也得到了修正坐标---现在又拿出来用了）将anchor变换成推荐区域（能怎么变？！不还是那些把框修修变成推荐区域嘛！）
RPN生成RoIs的过程(ProposalCreator)如下：

    对于每张图片，利用它的feature map， 计算 (H/16)× (W/16)×9（大概20000）个anchor属于前景的概率，以及对应的位置参数。
    选取概率较大的12000个anchor利用回归的位置参数，修正这12000个anchor的位置，得到RoIs利用非极大值（(Non-maximum suppression, NMS）抑制，选出概率最大的2000个RoIs
    
    https://blog.csdn.net/jiongnima/article/details/80478597
     在RPN中，从上万个anchor中，选择一定数目（2000或者300），-------？怎么又变成这么多个了！！调整大小和位置，生成RoIs，用以Fast R-CNN训练或者测试。
"""#函数的输入为rpn_cls_prob_reshape：rpn_cls_score经过R-softmax-R，rpn_bbox_pred:bbox信息预测结果
def proposal_layer(rpn_cls_prob_reshape,rpn_bbox_pred,im_info,cfg_key,_feat_stride = [16,],anchor_scales = [8, 16, 32]):
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_)
    _anchors = generate_anchors(scales=np.array(anchor_scales)) #生成9个锚点，shape: [9,4]
    _num_anchors = _anchors.shape[0] #num_anchors值为9
    rpn_cls_prob_reshape = np.transpose(rpn_cls_prob_reshape,[0,3,1,2])#将RPN输出的分类信息维度变成[N,C,H,W]
    rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,1,2]) #将RPN输出的边框变换信息维度变成[N,C,H,W]


    im_info = im_info[0]


    assert rpn_cls_prob_reshape.shape[0] == 1, \
        'Only single item batches are supported'
    # cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
    #cfg_key = 'TEST'
    #以下代码获取配置信息， 包括nms的信息
    pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
    min_size      = cfg[cfg_key].RPN_MIN_SIZE

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    scores = rpn_cls_prob_reshape[:, _num_anchors:, :, :]  #讲道理，这个reshape一下怎么就和score联系在了一起   #rpn_bbox_cls_prob另一个文件这里是这样的，而且整个大函数也是这个作为第一个参数
    # 重新取个变量名，bbox_deltas代表了RPN网络输出的各个框的变换信息
    bbox_deltas = rpn_bbox_pred
    #im_info = bottom[2].data[0, :]

    if DEBUG:
        print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print('scale: {}'.format(im_info[2]))

    # 1. Generate proposals from bbox deltas and shifted anchors
    #在这里得到了rpn输出的H和W
    height, width = scores.shape[-2:]

    if DEBUG:
        print('score map size: {}'.format(scores.shape))

    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

    # Convert anchors into proposals via bbox transformations通过bbox转换将anchor转换成proposals
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # 2. clip predicted boxes to image 在这里将超出图像边界的proposals进行边界裁剪，使之在图像边界之内
    proposals = clip_boxes(proposals, im_info[:2])#???

    # 3. remove predicted boxes with either height or width < threshold 去除那些宽度或者高度小于一定阈值的预测框，并返回符合条件的预测框的索引
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = _filter_boxes(proposals, min_size * im_info[2])#本文件函数
    #保留符合条件的proposals和scores
    proposals = proposals[keep, :]
    scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest按照score从大到小的顺序给（proposals，score）对进行排序
    # 5. take top pre_nms_topN (e.g. 6000)  取出score最高的pre_nms_topN个（proposals，score）对
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7) nms 非极大值过滤掉,保留概率大的也就是得分搞的，比如车的识别率0.9
    # 7. take after_nms_topN (e.g. 300) 对（proposals，score）对应用nms
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]
    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob
    #top[0].reshape(*(blob.shape))
    #top[0].data[...] = blob

    # [Optional] output scores blob
    #if len(top) > 1:
    #    top[1].reshape(*(scores.shape))
    #    top[1].data[...] = scores

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

