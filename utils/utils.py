from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    # wh1: anchor (2) // anchors: (3, 2)
    # wh2: target_wh (n_boxes, 2)
    # wh2.t(): (2, n_boxes)
    # e.g
    # wh1 = torch.Tensor(np.random.random((2)))
    # wh2 = torch.Tensor(np.random.random((2, 5)))
    # wh1: tensor([0.5401, 0.1042])
    # wh2: tensor([[0.6327, 0.2902, 0.6464, 0.3725, 0.8088],
    #              [0.2016, 0.0760, 0.9205, 0.3213, 0.7930]])
    # w1: tensor(0.5401)
    # h1: tensor(0.1042)
    # w2: tensor([0.6327, 0.2902, 0.6464, 0.3725, 0.8088])
    # h2: tensor([0.2016, 0.0760, 0.9205, 0.3213, 0.7930])
    # torch.min()
    # minw = torch.min(w1, w2)——>tensor([0.5401, 0.2902, 0.5401, 0.3725, 0.5401])
    # minh = torch.min(h1, h2)——>tensor([0.1042, 0.0760, 0.1042, 0.1042, 0.1042])
    # inter_area = minw * minh = tensor([0.0563, 0.0221, 0.0563, 0.0388, 0.0563])
    # stack后final res shape:
    # e.g.(3, 5) ——>(num_anchor, n_boxes)
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """
    generate masks & t·
    :param pred_boxes: 预测的bbox(0, 13) (b, num_anchor, grid_size, grid_size, 4) -> (b, 3, 13, 13, 4)
    :param pred_cls: 预测的类别概率(0, 1) (b, num_anchor, grid_size, grid_size, n_classes) -> (b, 3, 13, 13, 80)
    :param target: label(0, 1) (n_boxes, 6), 第二个维度有6个值，分别为: box所属图片在本batch中的index， 类别index， xc, yc, w, h
    :param anchors: tensor([[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]) (num_anchor, 2) -> (3, 2-)->aw, ah
    :param ignore_thres: hard coded, 0.5
    :return: masks & t·
    """

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)     # batch size
    nA = pred_boxes.size(1)     # anchor size: 3
    nC = pred_cls.size(-1)      # class size: 80
    nG = pred_boxes.size(2)     # grid size: 13

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)      # (b, 3, 13, 13)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)    # (b, 3, 13, 13)    # mostly candidates are noobj
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)   # (b, 3, 13, 13)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)   # (b, 3, 13, 13)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)           # (b, 3, 13, 13)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)           # (b, 3, 13, 13)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)           # (b, 3, 13, 13)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)           # (b, 3, 13, 13)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)     # (b, 3, 13, 13, 80)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG      # (0, 1)->(0, 13) (n_boxes, 4)
    gxy = target_boxes[:, :2]               # (n_boxes, 2)
    gwh = target_boxes[:, 2:]               # (n_boxes, 2)
    # Get anchors with best iou
    # 仅依靠w&h 计算target box和anchor box的交并比， (num_anchor, n_boxes)
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    # e.g, 每个anchor都和每个target bbox去算iou，结果存成矩阵(num_anchor, n_boxes)
    #                box0    box1    box2    box3    box4
    # ious=tensor([[0.7874, 0.9385, 0.5149, 0.0614, 0.3477],    anchor0
    #              [0.2096, 0.5534, 0.5883, 0.2005, 0.5787],    anchor1
    #              [0.8264, 0.6750, 0.4562, 0.2156, 0.7026]])   anchor2
    # best_ious:
    # tensor([0.8264, 0.9385, 0.5883, 0.2156, 0.7026])
    # best_n:
    # 属于第几个bbox：0, 1, 2, 3, 4
    #       tensor([2, 0, 1, 2, 2])   属于第几个anchor
    best_ious, best_n = ious.max(0)     # 最大iou, 与target box交并比最大的anchor的index // [n_boxes], [n_boxes]

    # Separate target values
    # target[:, :2]: (n_boxes, 2) -> img index, class index
    # target[].t(): (2, n_boxes) -> b: img index in batch, torch.Size([n_boxes]),
    #                               target_labels: class index, torch.Size([n_boxes])
    b, target_labels = target[:, :2].long().t()
    # gxy.t().shape = shape(gwh.t())=(2, n_boxes)
    gx, gy = gxy.t()            # gx = gxy.t()[0], gy = gxy.t()[1]
    gw, gh = gwh.t()            # gw = gwh.t()[0], gh = gwh.t()[1]
    gi, gj = gxy.long().t()     # .long()去除小数点
    # Set masks，这里的b是batch中的第几个
    obj_mask[b, best_n, gj, gi] = 1     # 物体中心点落在的那个cell中的，与target object iou最大的那个3个anchor中的那1个，被设成1
    noobj_mask[b, best_n, gj, gi] = 0   # 其相应的noobj_mask被设成0

    # Set noobj mask to zero where iou exceeds ignore threshold
    # ious.t():
    # shape: (n_boxes, num_anchor)
    # i: box id
    # b[i]: img index in that batch
    # E.g 假设有4个boxes，其所属图片在batch中的index为[0, 0, 1, 2], 即前2个boxes都属于本batch中的第0张图
    #     则b[0] = b[1] = 0 都应所属图片在batch中的index，即batch中的第几张图
    for i, anchor_ious in enumerate(ious.t()):
        # 如果anchor_iou>ignore_thres，则即使它不是obj(非best_n)，同样不算noobj
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()     # x_offset (0, 1)
    ty[b, best_n, gj, gi] = gy - gy.floor()     # y_offset (0, 1)
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    # 这两个是为了评价用，不参与实际回归
    # pred_cls与target_label匹配上了，则class_mask所对应的grid_xy为1，否则为0
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()        # target confidence
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
