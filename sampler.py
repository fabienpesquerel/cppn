import numpy as np
import imageio
import torch
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from model import CPPN
from mnist_data import MNIST
from torchvision.utils import save_image

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
parser.add_argument('--file_save', type=str, default='./images/test',
                    help='The file in which the output will be saved')
parser.add_argument('--frames', type=str, default=2,
                    help='number of frames in an animation for the gif')

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
        file_save = opt.file_save + '_' + str(i) + '.png'
        im = cppn.generator.generate_image(im)
        im = Image.fromarray(im)
        im.save(file_save)
        # save_image(im, file_save)


def save_anim_gif(filename=opt.file_save, n_frame=opt.frames,
                  duration1=0.5, duration2=1.0, duration=0.1,
                  number1=1, number2=8, number3=4,
                  scale1=4.0, scale2=17, reverse_gif=True):
    images = []
    enc = []
    n1 = True
    n2 = True
    n3 = True
    for idx, (im, label) in enumerate(mnist_train):
        if label.item() == number1 and n1:
            n1 = False
            if opt.cuda:
                im = im.cuda()
            mean, logvar = cppn.encoder(im)
            enc.append(cppn.reparametrize(mean, logvar))
        if label.item() == number2 and n2:
            n2 = False
            if opt.cuda:
                im = im.cuda()
            mean, logvar = cppn.encoder(im)
            enc.append(cppn.reparametrize(mean, logvar))
        if label.item() == number3 and n3:
            n3 = False
            if opt.cuda:
                im = im.cuda()
            mean, logvar = cppn.encoder(im)
            enc.append(cppn.reparametrize(mean, logvar))

    delta = []
    n = len(enc)
    for i in range(n-1):
        delta_z = (enc[i+1] - enc[i]) / (n_frame + 1)
        delta.append(delta_z)
    delta_s = (scale2 - scale1) / (n_frame + 1)
    s = scale1
    e = enc[0]
    frames = 0
    for i in range(n_frame):
        cppn.generator.scale = s
        im = cppn.generator.generate_image(e)
        # im = Image.fromarray(im)
        images.append(im)
        s += delta_s
        frames += 1
    print(images)
    durations = [duration1] + [duration]*(frames - 2) + [duration2]
    revImages = list(images)
    revImages.reverse()
    revImages = revImages[1:]
    images = images + revImages
    durations = durations + [duration] * (frames - 2) + [duration1]
    frames = 0
    images2 = []
    for j in range(len(delta)):
        z1 = enc[j]
        delta_z = delta[j]
        for i in range(n_frame):
            z_gen = z1 + delta_z*float(i)
            print("processing image ", i)
            im = cppn.generator.generate_image(z_gen)
            # im = Image.fromarray(im)
            images2.append(im)
            frames += 1
    durations2 = [duration] + [duration]*(frames - 2) + [duration2]
    if reverse_gif:
        rev = list(images2)
        rev.reverse()
        rev = rev[1:]
        images2 = images2 + rev
        durations2 = durations2 + [duration] * (frames - 2) + [duration1]

    images = images + images2
    durations = durations + durations2

    print("Writing a gif...")
    filename = filename + ".gif"
    imageio.mimsave(filename, images, duration=durations)


if opt.output == 'gif':
    save_anim_gif()
