from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    # module_defs = [{"type":"net", "channels":3, ...},                     # each elemnt is a layer block (dtype=dict)
    #                {"type":"convolutional", "batch_normalize":1, ...},
    #                ...]

    hyperparams = module_defs.pop(0)                    # [net]的整体参数
    output_filters = [int(hyperparams["channels"])]     # 3: 最初。因为是rgb 3通道
    module_list = nn.ModuleList()   # 存储每一大层，如conv层: 包括conv-bn-leaky relu等
    # nn.ModuleList() & nn.Sequential()
    # nn.ModuleList(): 就是Module的list，并没有实现forward函数(并没有实际执行的函数)，所以只是module的list，并不需要module之间的顺序
    #                  关系
    # nn.Sequential(): module的顺序执行。是实现了forward函数的，即会顺序执行其中的module，所以每个module的size必须匹配
    # 说的不错的链接：https://blog.csdn.net/watermelon1123/article/details/89954224
    #              https://zhuanlan.zhihu.com/p/64990232
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()   # 存下每一大层的执行，如conv层: 包括conv-bn-leaky relu等
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])    # 输出channel个数
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",     # a newer formatting method for python3.x, called f-string. Better than %s..
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":     # .cfg中有linear activation，说明linear啥也不干
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])  # channel个数相加，对应concat
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            # # mask: 6,7,8 / 3,4,5 / 0,1,2 <=> 小/中/大 feature map <=> 大/中/小 物体
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)] # 获得w & h
            anchors = [anchors[i] for i in anchor_idxs]     # len=3, 3 anchors per level, 以416为基准的
            # for mask: 6,7,8
            # [(116, 90), (156, 198), (373, 326)]
            num_classes = int(module_def["classes"])        # 80
            img_size = int(hyperparams["height"])           # 416
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)         # 存入每个大层，如conv对应conv-bn-leaky relu，的执行
        output_filters.append(filters)      # 每层的output filter size，即channel个数。最初是3，对应rgb 3通道

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
    # upsample的方式至少有两种：
    # interpolate & transpose convolution
    # interpolate逐渐主流，原因是transpose convolution可能产生chessboard式的伪影(alias)


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()
    # 仅占位用，啥也不干


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)     # 3
        self.num_classes = num_classes      # 80
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()        # binary cross entropy
        # loss = a1*l_reg + a2*l_conf + a3*l_cls
        # l_conf = obj_scale*l_obj + noobj_scale * l_noobj
        self.obj_scale = 1                  # lambda们
        self.noobj_scale = 100
        self.metrics = {}                   # 一堆计算变量
        self.img_dim = img_dim              # 图像大小，416
        self.grid_size = 0  # grid size     # 13x13=>32, 26x26=>16, 52x52=>8

    def compute_grid_offsets(self, grid_size, cuda=True):
        # 0<-13; 13<-26; 26<-52
        self.grid_size = grid_size
        g = self.grid_size          # 13, 26, 52
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size     # 32, 16, 8 => pixels per grid/feature point represents
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        # torch.arange(g): tensor([0,1,2,...,12])
        # torch.arange(g).repeat(g, 1):
        #       tensor([[0,1,2,...,12],
        #               [0,1,2,...,12],
        #               ...
        #               [0,1,2,...,12]])
        #       shape=torch.Size([13, 13])
        # torch.arange(g).repeat(g, 1).view([1, 1, g, g]):
        #       tensor([[[[0,1,2,...,12],
        #                 [0,1,2,...,12],
        #                 ...
        #                 [0,1,2,...,12]]]])
        #       shape=torch.Size([1, 1, 13, 13])
        # todo: 关于 repeat (不是todo，就是为了这个颜色)
        # torch.repeat(m): 在第0维重复m次
        #                  此处如果只用.repeat(g),则会出现[0,1,...,12,0,1,...12,...,0,1,...12]
        # torch.repeat(m, n): 在第0维重复m次，在第1维重复n次

        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        # torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]):
        #       tensor([[[[0,0,0,...,0],
        #                 [1,1,1,...,1],
        #                 ...
        #                 [12,12,12,...,12]]]])
        #       shape=torch.Size([1, 1, 13, 13])
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        # FloatTensor()后会将里面的tuple()变成[]
        # 将anchor变到(0, 13)范围内
        # self.scaled_anchors = tensor([[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]) # 3x2
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        # self.scaled_anchors[:, :1]: tensor([[3.625], [4.8750], [11.6562]])
        # self.anchor_w =
        # self.scaled_anchors.view((1, 3, 1, 1)) =
        #                                          tensor([
        #                                                  [
        #                                                    [[3.625]],
        #                                                    [[4.8750]],
        #                                                    [[11.6562]]
        #                                                  ]
        #                                                 ])
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        # x.shape: b x 255 x 13 x 13 (anchor 6, 7, 8)

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)     # batch size
        grid_size = x.size(2)       # feature map size: 13, 26, 52  # initially, self.grid_size = 0

        prediction = (
            #       b, 3, 85, 13, 13
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            #       b, 3, 13, 13, 85
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        # the x,y,w,h corresponds to the pink circle in slides (generated directly from network)
        x = torch.sigmoid(prediction[..., 0])  # Center x   # (b,3,13,13)            # 1 +
        y = torch.sigmoid(prediction[..., 1])  # Center y   # (b,3,13,13)            # 1 +
        w = prediction[..., 2]  # Width                     # (b,3,13,13)            # 1 +
        h = prediction[..., 3]  # Height                    # (b,3,13,13)            # 1 +
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf (b,3,13,13)            # 1 + = 5 +
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. (b,3,13,13,80)    # 80 = 85

        # Initially, self.grid_size = 0 != 13, then 13 != 26, then 26 != 52
        # Each time, if former grid size does not match current one, we need to compute new offsets
        # 作用：
        # 1. 针对不同size的feature map (13x13, 26x26, 52x52), 求出不同grid的左上角坐标
        # 2. 将(0, 416)范围的anchor scale到(0, 13)的范围
        #
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)
        # self.grid_x:                             # self.grid_y:
        #       tensor([[[[0,1,2,...,12],          #       tensor([[[[0,0,0,...,0],
        #                 [0,1,2,...,12],          #                 [1,1,1,...,1],
        #                 ...                      #                 ...
        #                 [0,1,2,...,12]]]])       #                 [12,12,12,...,12]]]])
        #       shape=torch.Size([1, 1, 13, 13])   #       shape=torch.Size([1, 1, 13, 13])
        #                                          #
        # self.anchor_w: shape([1, 3, 1, 1])       # self.anchor_h: shape([1, 3, 1, 1])
        # tensor([                                 # tensor([
        #         [                                #         [
        #           [[3.625]],                     #           [[2.8125]],
        #           [[4.8750]],                    #           [[6.1875]],
        #           [[11.6562]]                    #           [[10.1875]]
        #         ]                                #         ]
        #        ])                                #        ])

        # Add offset and scale with anchors
        # 请回想/对照slides中的等式，是目前绝大部分靠回归offset的方法通行的策略
        # x, y, w, h即上文中prediction, 对应t·,也即offset们, 此部分是直接由网络predict出来的, xy经过sigmoid强制到(0,1)
        # grid_xy是grid的左上角坐标[0,1,...,12],
        # 所以xy+grid_xy就是将pred结果(即物体中心点, slides中蓝色bx, by的部分)分布到每个grid中去，(0, 13)
        #
        # 对于wh，由于prediction的结果直接是log()后的(如果忘记，请回看slides：同样也对应蓝色bw,bh的部分)，所以此处要exp
        #
        # 此时，所有pred_boxes都是（0,13）范围的
        # These preds are final outpus for test/inference which corresponds to the blue circle in slides
        # This procedure could also be called as Decode
        #
        # 通常情况下，单纯的preds并不参与loss的计算，而只是作为最终的输出存在，
        # 但是这里依然计算，并在build_targets函数中出现，其目的，在于协助产生mask
        pred_boxes = FloatTensor(prediction[..., :4].shape)     # (b, 3, 13, 13, 4)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (   # * stride(=32对于13x13)，目的是将(0, 13)的bbox恢复到(0, 416)
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            # iou_scores: [b, num_anchor, grid_size, grid_size] -> pred_boxes与ground_truth的IoU
            # class_mask: [b, num_anchor, grid_size, grid_size], 预测正确的class 为true
            # obj_mask : [b, num_anchor, grid_size, grid_size] -> 1: 一定是正样本落在的地方(b_id, anchor_id, i, j)
            #                                                  -> 0: 一定不是正样本落在的地方
            # noobj_mask:  [b, num_anchor, grid_size, grid_size] -> 1: 一定是负样本落在的地方
            #                                                    -> 0: 不一定是正样本落在的地方，也可能是不参与计算
            #                                                          体现了ignore_thres的价值。>ignore的，都不参与计算
            # 底下是，算出来的，要参与产生loss的真实target.(除了tcls)
            # The procedure to generate those t·, corresponding to the gray circle in slides, can be called as Encode
            # tx: [b, num_anchor, grid_size, grid_size]
            # ty: [b, num_anchor, grid_size, grid_size]
            # tw: [b, num_anchor, grid_size, grid_size]
            # th: [b, num_anchor, grid_size, grid_size]
            # tcls :[b, num_anchor, grid_size, grid_size, n_classes]
            #
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,              # (b, 3, 13, 13, 4)
                pred_cls=pred_cls,                  # (b, 3, 13, 13, 80)
                target=targets,                     # (n_boxes, 6) [details in build_targets function]
                anchors=self.scaled_anchors,        # (3, 2) 3个anchor，每个2维
                ignore_thres=self.ignore_thres,     # 0.5 (hard code in YOLOLayer self.init())
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            # 可以看到，真正参与loss计算的，仍然是·与t·，即offset regress
            # Reg Loss
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

            # Conf Loss
            # 因为这里conf选择的是bce_loss，因为对于noobj，基本都能预测对，所以loss_conf_noobj通常比较小
            # 所以此时为了平衡，noobj_scale往往大于obj_scale, (100, 1)
            # 实际上，这里的conf loss就是做了个0-1分类，0就是noobj, 1就是obj
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj

            # Class Loss
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

            # Total Loss
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()     # class_mask/obj_mask(b, 3, 13, 13) # 正确率
            conf_obj = pred_conf[obj_mask].mean()           # 有物体的平均置信度
            conf_noobj = pred_conf[noobj_mask].mean()       # 无物体的平均置信度
            conf50 = (pred_conf > 0.5).float()              # 置信度大于0.5的位置 (b, num_anchor, 13, 13)
            iou50 = (iou_scores > 0.5).float()              # iou大于0.5的位置 (b, num_anchor, 13, 13)
            iou75 = (iou_scores > 0.75).float()             # iou大于0.75的位置 (b, num_anchor, 13, 13)
            detected_mask = conf50 * class_mask * tconf     # tconf=obj_mask, 即：既是预测的置信度>0.5，又class也对，又是obj
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        # Each element of the module_defs is a dict, a layer block with key values like 'type', 'batch_normalize', etc
        # module_defs = [{"type":"net", "channels":3, ...},         # each elemnt is a layer block (dtype=dict)
        #                {"type":"convolutional", "batch_normalize":1, ...},
        #                ...]
        self.module_defs = parse_model_config(config_path)      # read in cfg where net is defined
        # hyperparams: {"type":"net", "channels":3, ...}
        # module_list: 每个layer-block的顺序执行（不包含module_defs[0](也就是[net]layer的，那层是hyperparams)）
        #      create_modules中，为提取hyperparams，已pop出hyper，所以module_defs此时已无[net]module
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")] # not used
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        # x = b*3*416*416
        img_dim = x.shape[2]        # 416
        loss = 0
        layer_outputs, yolo_outputs = [], []        # 此时module_defs已无[0](net layer),是从conv开始的
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)   # conv(-batch-leakyrelu)非yolo之前的
            elif module_def["type"] == "route":
                #              layer_outputs就是每个module block的输出
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]  # element-wise addition
            elif module_def["type"] == "yolo":
                # module[0] here: YOLOLayer.forward
                # Because module_list here corresponds .add_module(..., YOLOLayer), and it's under nn.Sequential,
                # so we need excute the .forward function
                x, layer_loss = module[0](x, targets, img_dim)      # targets: ground truth, from dataloader
                # 此时x为predicted outputs
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
