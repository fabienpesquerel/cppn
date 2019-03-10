import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import IPython


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)


class CPPN(nn.Module):
    def __init__(self, batch_size=1, z_dim=32, c_dim=1,
                 scale=4.0, x_dim=28, y_dim=28, layers=4,
                 size=256, metric='2', activation='tanh', leak=None,
                 learning_rate=0.01, learning_rate_d=0.001, beta1=0.9,
                 learning_rate_vae=0.0001, net_size_q=512, keep_prob=1.0,
                 df_dim=24, model_name='cppn', net_size_g=128,
                 net_depth_g=4, cuda_device=None):
        super(CPPN, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.scale = scale
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.metric = metric
        self.learning_rate = learning_rate
        self.learning_rate_d = learning_rate_d
        self.learning_rate_vae = learning_rate_vae
        self.beta1 = beta1
        self.net_size_g = net_size_g
        self.net_size_q = net_size_q
        self.net_depth_g = net_depth_g
        self.model_name = model_name
        self.keep_prob = keep_prob
        self.df_dim = df_dim
        n_points = x_dim * y_dim
        self.n_points = n_points

        self.ones = torch.ones(batch_size, 1)
        self.zeros = torch.zeros(batch_size, 1)
        self.device = cuda_device
        if cuda_device is not None:
            self.ones = self.ones.cuda()
            self.zeros = self.zeros.cuda()

        self.encoder = Encoder()
        self.discriminator = Discriminator()
        self.generator = Generator(cuda_device=self.device,
                                   x_dim=x_dim,
                                   y_dim=y_dim,
                                   scale=scale)

        self.optimizer_encoder = optim.Adam(self.encoder.parameters(),
                                            lr=learning_rate_vae,
                                            weight_decay=1e-4)
        self.optimizer_generator = optim.Adam(self.generator.parameters(),
                                              lr=learning_rate,
                                              weight_decay=1e-4)
        self.optimizer_discriminator = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate_d,
            weight_decay=1e-4)
        self.scheduler_encoder = optim.lr_scheduler.StepLR(
            self.optimizer_encoder,
            step_size=1,
            gamma=0.9)
        self.scheduler_generator = optim.lr_scheduler.StepLR(
            self.optimizer_generator,
            step_size=1,
            gamma=0.9)
        self.scheduler_discriminator = optim.lr_scheduler.StepLR(
            self.optimizer_discriminator,
            step_size=1,
            gamma=0.9)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mean, logvar = self.encoder.forward(x)
        encoding = self.reparametrize(mean, logvar)
        decoding = self.generator.forward(encoding)
        d_real = self.discriminator.forward(x)
        d_fake = self.discriminator.forward(decoding.view(self.batch_size,
                                                          self.c_dim,
                                                          self.x_dim,
                                                          self.y_dim))
        return decoding, mean, logvar, d_real, d_fake

    def loss_encoder(self, reconstruction, target, mu, logvar):
        BCE = F.binary_cross_entropy(reconstruction, target,
                                     reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (BCE + KLD) / self.n_points, BCE

    def loss_discriminator(self, reconstruction_discriminator,
                           target_discriminator):
        loss_real = F.binary_cross_entropy_with_logits(target_discriminator,
                                                       self.ones)
        loss_fake = F.binary_cross_entropy_with_logits(
            reconstruction_discriminator,
            self.zeros)
        loss = loss_real + loss_fake
        return loss, loss_fake

    def loss_generator(self, loss_fake, ae_loss, BCE):
        return loss_fake + ae_loss + BCE / (2. * self.n_points)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 256)
        self.fc2_mean = nn.Linear(256, 32)
        self.fc2_std = nn.Linear(256, 32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_std(x)
        return mean, logvar


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class Generator(nn.Module):
    def __init__(self, batch_size=1, z_dim=32, c_dim=1,
                 scale=4.0, x_dim=28, y_dim=28, layers=4,
                 size=256, metric='2', activation='tanh', leak=None,
                 cuda_device=None):
        super(Generator, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.scale = scale
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.metric = metric
        self.device = cuda_device
        n_points = x_dim * y_dim
        self.n_points = n_points
        in_layer = 3 + z_dim

        x_mat, y_mat, r_mat = self._coordinates()
        self.x_unroll = x_mat.view(self.n_points, 1)
        self.y_unroll = y_mat.view(self.n_points, 1)
        self.r_unroll = r_mat.view(self.n_points, 1)

        model = [nn.Linear(in_layer, size)]

        for _ in range(layers):
            if activation == 'tanh':
                model += [nn.Linear(size, size,), nn.Tanh()]
            elif activation == 'relu':
                if leak is None:
                    model += [nn.Linear(size, size,), nn.LeakyReLU(0.01)]
                else:
                    model += [nn.Linear(size, size,), nn.LeakyReLU(leak)]

        model += [nn.Linear(size, c_dim), nn.Sigmoid()]

        self.generator = nn.Sequential(*model)

    """
    def generate_image(self, z=None):
        if z is None:
            z = torch.randn(self.batch_size, self.z_dim)
        z = z.double()
        x_mat, y_mat, r_mat = self._coordinates()

        x_unroll = x_mat.view(self.n_points)
        y_unroll = y_mat.view(self.n_points)
        r_unroll = r_mat.view(self.n_points)

        image = []
        with torch.no_grad():
            for pixel in range(self.n_points):
                coord = torch.tensor([x_unroll[pixel], y_unroll[pixel],
                                      r_unroll[pixel]]).double()

                x = torch.cat((coord, z[0])).float()

                intensity = self.generator(x)
                if self.c_dim > 1:
                    intensity = intensity.numpy().reshape(self.c_dim)
                image.append(intensity)
        image = (255.0 * np.array([image]))
        if self.c_dim > 1:
            image = np.array(image.reshape(self.y_dim,
                                           self.x_dim, self.c_dim),
                             dtype=np.uint8)
        else:
            image = np.array(image.reshape(self.y_dim,
                                           self.x_dim), dtype=np.uint8)
        return image
    """

    def generate_image(self, z=None):
        if z is None:
            z = torch.randn(self.batch_size, self.z_dim)
        z = z.double()
        x_unroll = self.x_unroll
        y_unroll = self.y_unroll
        r_unroll = self.r_unroll
        coord = torch.cat((x_unroll, y_unroll), dim=1)
        coord = torch.cat((coord, r_unroll), dim=1)
        if self.device is not None:
            coord = coord.cuda()
        # Coord has dim n_points * 3 at this point
        z = z.view(1, self.z_dim)
        z = z.expand(self.n_points, self.z_dim)
        x = torch.cat((coord, z), dim=1).float()
        intensity = self.generator(x)
        image = intensity.numpy()
        image = np.array(image.reshape(self.y_dim,
                                       self.x_dim), dtype=np.uint8)
        return image

    def forward(self, z):
        z = z.double()
        x_unroll = self.x_unroll
        y_unroll = self.y_unroll
        r_unroll = self.r_unroll
        coord = torch.cat((x_unroll, y_unroll), dim=1)
        coord = torch.cat((coord, r_unroll), dim=1)
        if self.device is not None:
            coord = coord.cuda()
        # Coord has dim n_points * 3 at this point
        z = z.view(1, self.z_dim)
        z = z.expand(self.n_points, self.z_dim)
        x = torch.cat((coord, z), dim=1).float()
        intensity = self.generator(x)
        return intensity

    def _coordinates(self):
        x_dim = self.x_dim
        y_dim = self.y_dim
        scale = self.scale
        n_points = x_dim * y_dim
        x_range = scale * (np.arange(x_dim) - (x_dim - 1)/2.0)/(x_dim - 1)/0.5
        y_range = scale * (np.arange(y_dim) - (y_dim - 1)/2.0)/(y_dim - 1)/0.5
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))

        if self.metric == '1':
            r_mat = np.abs(x_mat) + np.abs(y_mat)
        elif self.metric == '2':
            r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
        elif self.metric == 'inf':
            r_mat = (x_mat + y_mat)/2. + np.abs(x_mat - y_mat)/2.
        else:
            r_mat = np.power(x_mat,
                             int(self.metric)) + np.power(y_mat,
                                                          int(self.metric))
            r_mat = np.power(r_mat, 1./float(self.metric))

        x_mat = np.tile(x_mat.flatten(),
                        self.batch_size).reshape(self.batch_size, n_points, 1)
        y_mat = np.tile(y_mat.flatten(),
                        self.batch_size).reshape(self.batch_size, n_points, 1)
        r_mat = np.tile(r_mat.flatten(),
                        self.batch_size).reshape(self.batch_size, n_points, 1)
        return torch.tensor(x_mat), torch.tensor(y_mat), torch.tensor(r_mat)

    def reinit(self):
        self.apply(weights_init_normal)
