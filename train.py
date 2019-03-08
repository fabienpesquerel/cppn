import torch
from mnist_data import MNIST
from model import CPPN
from torchvision.utils import save_image
import torch.nn.functional as F
import argparse
# from PIL import Image

parser = argparse.ArgumentParser()


parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=4,
                    help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the\
 batches')
parser.add_argument('--cuda', action='store_true',
                    help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4,
                    help='number of cpu threads to use during\
 batch generation')

opt = parser.parse_args()

print(opt)


mnist = MNIST()

train_mnist = mnist.train_loader
test = mnist.test_loader
model = CPPN()

if opt.cuda:
    model.cuda()

le = 0
ld = 0
lg = 0

model.train()
for epoch in range(opt.n_epochs):
    print("STARTING EPOCH {}".format(epoch))
    for idx, (im, _) in enumerate(train_mnist):
        model.optimizer_discriminator.zero_grad()
        model.optimizer_encoder.zero_grad()
        model.optimizer_generator.zero_grad()

        if opt.cuda:
            im.cuda()
            print(im.device())

        gen, mu, logvar, d_r, d_f = model.forward(im)

        loss_encoder = model.loss_encoder(gen, im.view(model.n_points,
                                                       model.batch_size),
                                          mu, logvar)
        loss_discriminator, l_f = model.loss_discriminator(d_f, d_r)
        loss_generator = model.loss_generator(l_f, loss_encoder)

        loss_encoder.backward(retain_graph=True)
        loss_discriminator.backward(retain_graph=True)
        loss_generator.backward()

        model.optimizer_discriminator.step()
        model.optimizer_encoder.step()
        model.optimizer_generator.step()

        le += loss_encoder.item()
        ld += loss_discriminator.item()
        lg += loss_generator.item()

        if idx % 501 == 0:
            print("loss encoder: {}".format(le / 501.))
            print("loss discriminator: {}".format(ld / 501.))
            print("loss generator: {}".format(lg / 501.))
            le = 0
            ld = 0
            lg = 0

        if idx % 10000 == 0:
            im_save = gen.view(model.c_dim,
                               model.x_dim,
                               model.y_dim)
            save_image(im_save, 'im_' + str(epoch) + '_' + str(idx) + '.png')
            save_image(im_save, 'im_true_' + str(epoch) + '_' +
                       str(idx) + '.png')

    torch.save(model.state_dict(), './model_' + str(epoch) + 'pt')
    model.scheduler_discriminator.step()
    model.scheduler_encoder.step()
    model.scheduler_generator.step()

torch.save(model.state_dict(), './model_final.pt')
