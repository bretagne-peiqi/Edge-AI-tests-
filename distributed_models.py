import argparse
import collections as co
import torch
import os

import torch.nn as nn
from torch.nn import Module

class Detect(nn.Module):
    def __init__(self, nc=80, anchors=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.export = False  # onnx export

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


# FIXME(Peiqi)
# the idea is to load corresponding state_dict weights to its model.
# to complicate that, we need config automatically model config files and split weights according to its model. 
def ModelSplit():

    ModelPath = opt.models
    Weights = opt.weights
    new_weights_dict = co.OrderedDict()
    splitN = opt.splitN
    savePath=opt.savePath
    model = torch.load(ModelPath) 
    ckpt = torch.load(Weights)

    try:
        weights = {k: v for k, v in ckpt['model'].float().state_dict().items()
                    if int(k.split('.')[1]) >= splitN }

        print ('weights is ', len(weights.keys()))
        print ('model.state_dict is ', len(model.state_dict().keys()))
        c = 0
        for i in weights.keys(): 
            for j in model.state_dict().keys():
                if 'model.'+ j == i:
                    c = c + 1
                    print (i, j)

        print ('count ', c)

        #model.state_dict().update(weights)
        #for k, v in weights.items():
        #   name = k[6:]
        #   new_weights_dict[name] = v

        for k, v in zip(model.state_dict().keys(), weights.values()):
            new_weights_dict[k] = v

    
        
        #print ('before model is ', model.state_dict().values())
        #model.load_state_dict(new_weights_dict, strict=True)
        #print ('after model is ', model.state_dict().values())
        #savePath = opt.savePath+i+'.pt'
        #torch.save(model, savePath)
    except KeyError as e:
        s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s." \
            % (opt.models, opt.weights)
        raise KeyError(s) from e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, nargs='+', action='append', default='weights/yolov5s.pt', help='pretrained model weights path')
    parser.add_argument('--models', type=str, nargs='+', action='append', default='models/wei2.pt', help='segmented model slices path')
    parser.add_argument('--savePath', type=str, default='weights/gwei2.pt', help='path to output segmented model slices')
    parser.add_argument('--splitN', default=0, help='split model segmentation from number N')
    opt = parser.parse_args()

    with torch.no_grad():
        ModelSplit()
