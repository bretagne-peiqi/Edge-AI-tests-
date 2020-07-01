import argparse
import collections as col

from models.yolo import Model
import torch.nn as nn
import torch
from utils.utils import *

#loaded_dict = {k: loaded_dict[k] for k, _ in model.state_dict()}
def split():
    device = torch_utils.select_device(opt.device)

    model = torch.load('/home/edge/peiqi/wei.pt', map_location=device)  # load to FP32
    print(model.state_dict().keys())
    print ('before\n', model.state_dict()['1.conv.weight'])

    weights = opt.weights
    splitN = opt.splitN
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    new_state_dict = col.OrderedDict()

    # load model
    try:
        dicter = {k: v for k, v in ckpt['model'].float().state_dict().items()#}
                           if int(k.split('.')[1]) < splitN}  # to FP32, filter
        print ('good weights: \n', (dicter['model.1.conv.weight']))

        for k, v in dicter.items():
            name = k[6:] # remove `module.`
            new_state_dict[name] = v
        #print ('good weights: \n', (new_state_dict['0.conv.conv.weight']))

        model.load_state_dict(new_state_dict, strict=True)
        print ('final \n', model.state_dict()['1.conv.weight'])
    except KeyError as e:
        s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s." \
            % (opt.weights, opt.cfg, opt.weights)
        raise KeyError(s) from e

    #print ('after\n', model.state_dict().items())
    torch.save(model, '/home/edge/peiqi/weih.pt')
    ##for k, v in enumerate(model.state_dict()):
     #   if k <= opt.splitN:
      #      dict1[k] = v
      #      continue
      #  dict2[k] = v###
    #for m in model.parameters():
        #print ("m is: ", m)
     
        #continue
    #for m, _ in model.named_parameters():
       # print ("m and p are: ", m)
        #continue
    ##for idx, n in model.named_modules():
      ##  print (idx, " ---> ", n)
    #print (len(model.state_dict()))

    #model2[]
    #torch.save(dict1, opt.path)

    #model2 = torch.load('/home/edge/peiqi/distYolov5/wei2.pt', map_location=device)['state_dict'].float()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--splitN', default=3, help='split model segmentation from number N')
    parser.add_argument('--cfg', type=str, default='models/yolov5s1.yaml', help='*.cfg path')
    opt = parser.parse_args()

    with torch.no_grad():
        split()

