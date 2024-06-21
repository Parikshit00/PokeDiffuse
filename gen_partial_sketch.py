#!/usr/bin/env python

import getopt
import numpy
import PIL
import PIL.Image
import sys
import torch
import os
import cv2
import numpy as np
import random
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = False

args_strModel = 'bsds500' # only 'bsds500' for now
args_strIn = './static/sample2.jpg'
args_strOut = './out.png'
args_cuda = False

for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
    'model=',
    'in=',
    'out=',
    'cuda=',
])[0]:
    if strOption == '--model' and strArg != '': args_strModel = strArg # which model to use
    if strOption == '--in' and strArg != '': args_strIn = strArg # path to the input image
    if strOption == '--out' and strArg != '': args_strOut = strArg # path to where the output should be stored
    if strOption == '--cuda' and strArg != '': args_cuda = strArg # whether to use CUDA
# end
netNetwork = None

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-hed/network-' + args_strModel + '.pytorch', file_name='hed-' + args_strModel).items() })
    # end

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(data=[104.00698793, 116.66876762, 122.67891434], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))

def estimate(tenInput, use_cuda):
    global netNetwork

    if netNetwork is None:
        if use_cuda:
            torch.backends.cudnn.enabled = True
            netNetwork = Network().cuda().eval()
        else:
            netNetwork = Network().cpu().eval()
    print(tenInput.shape)
    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[1]
    assert(intWidth == 480)
    assert(intHeight == 320)
    if use_cuda:
        netNetwork(tenInput.cuda().view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()
    else:
        return netNetwork(tenInput.cpu().view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()

def random_delete(contour_image, n_b, w_b, h_b, delete_probability=0.7):
    height, width = contour_image.shape
    for _ in range(n_b):
        if random.random() < delete_probability:
            x = random.randint(0, width - w_b)
            y = random.randint(0, height - h_b)
            contour_image[y:y+h_b, x:x+w_b] = 255
    return contour_image

def hed2sketch(edges_image, out_sketch):
    _, edges_image = cv2.threshold(edges_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(edges_image)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
    contour_image = cv2.bitwise_not(contour_image)
    print("Getting contour image...")
    print(contour_image.shape)
    n_b = 150
    w_b = 30
    h_b = 30
    delete_probability = 0.5  # 50% chance to delete each block
    contour_image = random_delete(contour_image, n_b, w_b, h_b, delete_probability)
    cv2.imwrite(out_sketch, contour_image)

if __name__ == '__main__':
    in_img = PIL.Image.open(args_strIn)
    in_img = in_img.resize((480, 320))
    tenInput = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(in_img)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    if args_cuda == 'True':
        args_cuda = True
    tenOutput = estimate(tenInput, args_cuda)
    img_array = (tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)
    hed = PIL.Image.fromarray(img_array)
    hed.save(args_strOut)
    filename, extension = os.path.splitext(os.path.basename(args_strOut))
    new_filename = filename + "_sketch" + extension
    opencv_image = cv2.imread(args_strOut, cv2.IMREAD_GRAYSCALE)
    hed2sketch(opencv_image, new_filename)
    
    
# end