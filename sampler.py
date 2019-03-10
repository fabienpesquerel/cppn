import numpy as np
import imageio
import torch
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from model import CPPN
from mnist_data import MNIST

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='./model_final.pt',
                    help='The model to be loaded')
parser.add_argument('--x_dim', type=int, default=1080,
                    help='Number of pixels in the x dimension')
parser.add_argument('--y_dim', type=int, default=1080,
                    help='Number of pixels in the y dimension')
parser.add_argument('--scale', type=float, default=5.0,
                    help='The scale is a zoom-in/zoom-out parameter')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--output', type=str, default='png',
                    help='Type of output: gif or png image')
parser.add_argument('--file_save', type=str, default='test',
                    help='The file in which the output will be saved')

opt = parser.parse_args()
print(opt)

mnist = MNIST()

mnist_train = mnist.train_loader

if opt.cuda:
    cuda_gpu = torch.device('cuda:0')
    cppn = CPPN(x_dim=opt.x_dim, y_dim=opt.y_dim, scale=opt.scale,
                cuda_device=cuda_gpu)
    cppn.cuda()
    cppn.load_state_dict(torch.load(opt.model))
else:
    cppn = CPPN(x_dim=opt.x_dim, y_dim=opt.y_dim, scale=opt.scale)
    cppn.load_state_dict(torch.load(opt.model, map_location='cpu'))

cppn.eval()


enc = []
lab = []
for idx, (im, label) in enumerate(mnist_train):
    with torch.no_grad():
        if opt.cuda:
            im = im.cuda()
        mean, logvar = cppn.encoder(im)
        encoding = cppn.reparametrize(mean, logvar)
    enc.append(encoding)
    lab.append(lab)
    if idx > 5:
        break


if opt.file_save == 'test':
    print('WARNING - The name of the file in which the output will be stored \
is still the one by default (test.gif or test.png).')

if opt.output == 'png':
    for i in range(len(enc)):
        im = enc[i]
        la = lab[i]
        file_save = opt.file_save + '_' + str(i) + '_' + str(la) + '.png'
        im = cppn.generator.generate_image(im)
        im = Image.fromarray(im)
        im.save(file_save)
