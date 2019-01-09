#!/usr/bin/env python
# ------------------------------------------------------------------------
#
# Experimenting regression losses.
#
# Author Tuan Le (tuan.t.lei@gmail.com)
#
# ------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
import time
import copy
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms

import seaborn as sns
from matplotlib import pyplot

# ------------------------------------------------------------------------

DEVICE = 'cpu'

# ------------------------------------------------------------------------

np.random.seed(2)
torch.manual_seed(2)

sns.set(context='notebook', style='white', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)

# ------------------------------------------------------------------------


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        # return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)
        return torch.mean(2 * ey_t * torch.sigmoid(ey_t) - ey_t)


class AlgebraicLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * ey_t / torch.sqrt(1 + ey_t * ey_t))


class MNISTAutoencoderModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self._monitor = {
            'elapse_total_ms': 0,
            'learning': {
                'losses': []
            },
            'testing': {
                'losses': []
            }
        }
        self._scheduler = None
        self._optim = None
        self._criterion = None
        (self._encoder_fc, self._latent_fc, self._decoder_fc) = self._construct()

    def forward(self, x_t):
        x_t = self._encoder_fc(x_t)
        x_t = self._latent_fc(x_t)
        x_t = self._decoder_fc(x_t)
        return x_t

    @staticmethod
    def _construct():
        encoder_seqs = []
        latent_seqs = []
        decoder_seqs = []

        encoder_fclayer1 = torch.nn.Linear(in_features=28 * 28, out_features=256)
        torch.nn.init.xavier_normal_(encoder_fclayer1.weight)
        encoder_fclayer1.bias.data.fill_(0.0)
        encoder_seqs.append(encoder_fclayer1)
        encoder_seqs.append(torch.nn.ReLU(inplace=True))

        encoder_fclayer2 = torch.nn.Linear(in_features=256, out_features=64)
        torch.nn.init.xavier_normal_(encoder_fclayer2.weight)
        encoder_fclayer2.bias.data.fill_(0.0)
        encoder_seqs.append(encoder_fclayer2)
        encoder_seqs.append(torch.nn.ReLU(inplace=True))

        latent_fclayer1 = torch.nn.Linear(in_features=64, out_features=64)
        torch.nn.init.xavier_normal_(latent_fclayer1.weight)
        latent_fclayer1.bias.data.fill_(0.0)
        latent_seqs.append(latent_fclayer1)
        latent_seqs.append(torch.nn.BatchNorm1d(num_features=64))
        latent_seqs.append(torch.nn.ReLU(inplace=True))

        decoder_fclayer2 = torch.nn.Linear(in_features=64, out_features=256)
        torch.nn.init.xavier_normal_(decoder_fclayer2.weight)
        decoder_fclayer2.bias.data.fill_(0.0)
        decoder_seqs.append(decoder_fclayer2)
        decoder_seqs.append(torch.nn.ReLU(inplace=True))

        decoder_fclayer1 = torch.nn.Linear(in_features=256, out_features=28 * 28)
        torch.nn.init.xavier_normal_(decoder_fclayer1.weight)
        decoder_fclayer1.bias.data.fill_(0.0)
        decoder_seqs.append(decoder_fclayer1)
        decoder_seqs.append(torch.nn.Tanh())

        return (torch.nn.Sequential(*encoder_seqs), torch.nn.Sequential(*latent_seqs), torch.nn.Sequential(*decoder_seqs))

    @property
    def monitor(self):
        return copy.deepcopy(self._monitor)

    def setup(self, *, criterion='mse', optim='adam', lr=1e-3):
        if isinstance(optim, str):
            if optim == 'sgd':
                self._optim = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.0)
            elif optim == 'sgdm':
                self._optim = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
            elif optim == 'adam':
                self._optim = torch.optim.Adam(self.parameters(), lr=lr)
            else:
                raise TypeError('Unknown optimizer type %s.' % optim)
        else:
            self._optim = optim

        if isinstance(criterion, str):
            if criterion == 'mse' or criterion == 'mean_square_error':
                self._criterion = torch.nn.MSELoss(reduction='mean')
            elif criterion == 'mae' or criterion == 'mean_absolute_error':
                self._criterion = torch.nn.L1Loss(reduction='mean')
            elif criterion == 'xtl' or criterion == 'xtanh_loss':
                self._criterion = XTanhLoss()
            elif criterion == 'xsl' or criterion == 'xsigmoid_loss':
                self._criterion = XSigmoidLoss()
            elif criterion == 'agl' or criterion == 'algebraic_loss':
                self._criterion = AlgebraicLoss()
            elif criterion == 'lcl' or criterion == 'log_cosh_loss':
                self._criterion = LogCoshLoss()
            else:
                raise TypeError('Unknown criterion type %s.' % criterion)
            self._scheduler = torch.optim.lr_scheduler.StepLR(self._optim, step_size=5, gamma=0.9)
        else:
            self._criterion = criterion

    def infer(self, x_t):
        if isinstance(x_t, np.ndarray):
            x_t = torch.from_numpy(x_t).float().to(DEVICE)
        else:
            x_t = x_t.to(DEVICE)
        with torch.no_grad():
            y_t = self(x_t)
        return y_t

    def learn(self, x_t, y_prime_t, *,
              epoch_limit=50,
              batch_size=32,
              tl_split=0.2):
        if isinstance(x_t, np.ndarray):
            x_t = torch.from_numpy(x_t).float().to(DEVICE)
        else:
            x_t = x_t.to(DEVICE)

        if isinstance(y_prime_t, np.ndarray):
            y_prime_t = torch.from_numpy(y_prime_t).float().to(DEVICE)
        else:
            y_prime_t = y_prime_t.to(DEVICE)

        input_sample_size = x_t.shape[0]
        expected_output_sample_size = y_prime_t.shape[0]

        if input_sample_size != expected_output_sample_size:
            warnings.warn('Input training dataset is not the same lenght as the expected output dataset.', UserWarning)

        self._monitor['elapse_total_ms'] = 0
        self._monitor['learning']['losses'] = []
        self._monitor['testing']['losses'] = []

        if tl_split < 0 or tl_split > 0.5:
            tl_split = 0
            warnings.warn('Testing and learning split ratio must be >= 0 and <= 0.5. Reset testing and learning split ratio to default value of 0.', UserWarning)

        enable_testing = tl_split > 0

        if enable_testing:
            if input_sample_size == 1:
                learning_sample_size = input_sample_size
                enable_testing = False
                warnings.warn('Input sample size = 1. Reset testing and learning split ratio to default value of 0.', UserWarning)
            else:
                learning_sample_size = int(input_sample_size * (1 - tl_split))
                learning_sample_size = learning_sample_size - learning_sample_size % batch_size
                # testing_sample_size = input_sample_size - learning_sample_size
        else:
            learning_sample_size = input_sample_size

        if batch_size < 1 or batch_size > learning_sample_size:
            batch_size = learning_sample_size
            warnings.warn('Batch size must be >= 1 and <= learning sample size %d. Set batch size = learning sample size.' % learning_sample_size, UserWarning)

        elapse_total_ms = 0
        for epoch in range(epoch_limit):
            tstart_us = time.process_time()

            learning_loss_value = 0
            testing_loss_value = 0

            for i in range(learning_sample_size):
                if (i + batch_size) < learning_sample_size:
                    batched_x_t = x_t[i: i + batch_size].requires_grad_(True)
                    batched_y_prime_t = y_prime_t[i: i + batch_size].requires_grad_(False)
                # else:
                #     batched_x_t = x_t[i: learning_sample_size].requires_grad_(True)
                #     batched_y_prime_t = y_prime_t[i: learning_sample_size].requires_grad_(False)

                batched_y_t = self(batched_x_t)
                learning_loss_t = self._criterion(batched_y_t, batched_y_prime_t)

                self._optim.zero_grad()
                learning_loss_t.backward()
                self._optim.step()

                learning_loss_value += learning_loss_t.item()
            learning_loss_value /= learning_sample_size
            self._monitor['learning']['losses'].append(learning_loss_value)

            if enable_testing:
                with torch.no_grad():
                    batched_x_t = x_t[learning_sample_size:].requires_grad_(False)
                    batched_y_prime_t = y_prime_t[learning_sample_size:].requires_grad_(False)

                    batched_y_t = self(batched_x_t)
                    testing_loss_t = self._criterion(batched_y_t, batched_y_prime_t)

                    testing_loss_value = testing_loss_t.item()
                    self._monitor['testing']['losses'].append(testing_loss_value)

            tend_us = time.process_time()
            elapse_per_pass_ms = int(round((tend_us - tstart_us) * 1000))
            elapse_total_ms += elapse_per_pass_ms
            self._monitor['elapse_total_ms'] = elapse_total_ms

            if self._scheduler is not None:
                self._scheduler.step()
                lr = self._scheduler.get_lr()[0]
                print('Learning Rate: {lr:.9f}'.format(lr=lr))

            if enable_testing:
                print('Learning Epoch: {epoch}/{epoch_limit} - Elapse/Pass: {elapse_per_pass} ms - Elapse: {elapse_total} s - Learning Loss: {lloss:.9f} - Testing Loss: {tloss:.9f}'.format(
                    epoch=epoch + 1,
                    epoch_limit=epoch_limit,
                    elapse_per_pass=elapse_per_pass_ms,
                    elapse_total=round(elapse_total_ms * 0.001),
                    lloss=learning_loss_value,
                    tloss=testing_loss_value),
                    end='\n',  # end='\r',
                    flush=True)
            else:
                print('Learning Epoch: {epoch}/{epoch_limit} - Elapse/Pass: {elapse_per_pass} ms - Elapse: {elapse_total} s - Learning Loss: {lloss:.9f}'.format(
                    epoch=epoch + 1,
                    epoch_limit=epoch_limit,
                    elapse_per_pass=elapse_per_pass_ms,
                    elapse_total=round(elapse_total_ms * 0.001),
                    lloss=learning_loss_value),
                    end='\n',  # end='\r',
                    flush=True)


def run_training_adam(*,
                      train_cntl=True,
                      train_exp_lcl=True,
                      train_exp_xtl=True,
                      train_exp_xsl=True,
                      train_exp_agl=True):
    print('Training with Adam optimizer.\n')

    exp_model = MNISTAutoencoderModel().to(DEVICE)
    cntl_model = MNISTAutoencoderModel().to(DEVICE)
    monitors = {}

    epoch_limit = 50
    batch_size = 32
    tl_split = 0.2
    ds_sample_size = 512
    lr = 2e-3

    if not os.path.isfile('models/initial_state.pth'):
        torch.save(MNISTAutoencoderModel().to(DEVICE).state_dict(), 'models/initial_state.pth')
    initial_state = torch.load('models/initial_state.pth')

    ds_loader = DataLoader(datasets.MNIST('datasets',
                                          train=True, download=True,
                                          transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                           batch_size=ds_sample_size, shuffle=False)

    for ds in ds_loader:
        input_t = ds[0].to(DEVICE)
        input_t = input_t.view(input_t.size(0), -1)
        expected_output_t = input_t
        break

    if train_cntl:
        print('Training cntl model with MSE...\n')
        cntl_model.load_state_dict(initial_state)
        cntl_model.setup(criterion='mse', optim='adam', lr=lr)
        cntl_model.learn(input_t, expected_output_t, epoch_limit=epoch_limit, batch_size=batch_size, tl_split=tl_split)
        torch.save(cntl_model.state_dict(), 'models/cntl/trained_cntl_mse_adam.pth')
        with open('results/cntl/monitor_mse_adam.json', 'w') as monitor_json_file:
            json.dump({
                'learning': cntl_model.monitor['learning'],
                'testing': cntl_model.monitor['testing']
            }, monitor_json_file)
        monitors['mse'] = cntl_model.monitor
        print('Training cntl model with MAE...\n')
        cntl_model.load_state_dict(initial_state)
        cntl_model.setup(criterion='mae', optim='adam', lr=lr)
        cntl_model.learn(input_t, expected_output_t, epoch_limit=epoch_limit, batch_size=batch_size, tl_split=tl_split)
        torch.save(cntl_model.state_dict(), 'models/cntl/trained_cntl_mae_adam.pth')
        with open('results/cntl/monitor_mae_adam.json', 'w') as monitor_json_file:
            json.dump({
                'learning': cntl_model.monitor['learning'],
                'testing': cntl_model.monitor['testing']
            }, monitor_json_file)
        monitors['mae'] = cntl_model.monitor
    if train_exp_lcl:
        print('Training exp model with Log-Cosh Loss...\n')
        exp_model.load_state_dict(initial_state)
        exp_model.setup(criterion='lcl', optim='adam', lr=lr)
        exp_model.learn(input_t, expected_output_t, epoch_limit=epoch_limit, batch_size=batch_size, tl_split=tl_split)
        torch.save(exp_model.state_dict(), 'models/exp/trained_exp_lcl_adam.pth')
        with open('results/exp/monitor_lcl_adam.json', 'w') as monitor_json_file:
            json.dump({
                'learning': exp_model.monitor['learning'],
                'testing': exp_model.monitor['testing']
            }, monitor_json_file)
        monitors['lcl'] = exp_model.monitor
    if train_exp_xtl:
        print('Training exp model with XTanh Loss...\n')
        exp_model.load_state_dict(initial_state)
        exp_model.setup(criterion='xtl', optim='adam', lr=lr)
        exp_model.learn(input_t, expected_output_t, epoch_limit=epoch_limit, batch_size=batch_size, tl_split=tl_split)
        torch.save(exp_model.state_dict(), 'models/exp/trained_exp_xtl_adam.pth')
        with open('results/exp/monitor_xtl_adam.json', 'w') as monitor_json_file:
            json.dump({
                'learning': exp_model.monitor['learning'],
                'testing': exp_model.monitor['testing']
            }, monitor_json_file)
        monitors['xtl'] = exp_model.monitor
    if train_exp_xsl:
        print('Training exp model with XSigmoid Loss...\n')
        exp_model.load_state_dict(initial_state)
        exp_model.setup(criterion='xsl', optim='adam', lr=lr)
        exp_model.learn(input_t, expected_output_t, epoch_limit=epoch_limit, batch_size=batch_size, tl_split=tl_split)
        torch.save(exp_model.state_dict(), 'models/exp/trained_exp_xsl_adam.pth')
        with open('results/exp/monitor_xsl_adam.json', 'w') as monitor_json_file:
            json.dump({
                'learning': exp_model.monitor['learning'],
                'testing': exp_model.monitor['testing']
            }, monitor_json_file)
        monitors['xsl'] = exp_model.monitor
    if train_exp_agl:
        print('Training exp model with Algebraic Loss...\n')
        exp_model.load_state_dict(initial_state)
        exp_model.setup(criterion='agl', optim='adam', lr=lr)
        exp_model.learn(input_t, expected_output_t, epoch_limit=epoch_limit, batch_size=batch_size, tl_split=tl_split)
        torch.save(exp_model.state_dict(), 'models/exp/trained_exp_agl_adam.pth')
        with open('results/exp/monitor_agl_adam.json', 'w') as monitor_json_file:
            json.dump({
                'learning': exp_model.monitor['learning'],
                'testing': exp_model.monitor['testing']
            }, monitor_json_file)
        monitors['agl'] = exp_model.monitor

    figure1 = pyplot.figure()
    figure1.suptitle('Evaluations - Learning  Losses', fontsize=16)
    pyplot.xscale('linear')
    pyplot.yscale('linear')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    for (name, monitor) in monitors.items():
        pyplot.plot(monitor['learning']['losses'], linewidth=1, linestyle='solid', label='{name} - Learning Loss'.format(name=name))
    pyplot.legend(fancybox=True)
    pyplot.grid()

    figure2 = pyplot.figure()
    figure2.suptitle('Evaluations - Testing  Losses', fontsize=16)
    pyplot.xscale('linear')
    pyplot.yscale('linear')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    for (name, monitor) in monitors.items():
        pyplot.plot(monitor['testing']['losses'], linewidth=1, linestyle='solid', label='{name} - Testing Loss'.format(name=name))
    pyplot.legend(fancybox=True)
    pyplot.grid()
    pyplot.show()


def run_cmp_prediction_adam():
    print('Comparing prediction accuracy between EXP & CNTL with vanilla Adam optimizer.\n')

    batch_size = 32
    model_cntl_mse = MNISTAutoencoderModel().to(DEVICE)
    model_cntl_mae = MNISTAutoencoderModel().to(DEVICE)
    model_xtl_exp = MNISTAutoencoderModel().to(DEVICE)
    model_xsl_exp = MNISTAutoencoderModel().to(DEVICE)
    # model_agl_exp = MNISTAutoencoderModel().to(DEVICE)
    model_lcl_exp = MNISTAutoencoderModel().to(DEVICE)

    checkpoint = torch.load('models/cntl/trained_cntl_mse_adam.pth')
    model_cntl_mse.load_state_dict(checkpoint)

    checkpoint = torch.load('models/cntl/trained_cntl_mae_adam.pth')
    model_cntl_mae.load_state_dict(checkpoint)

    checkpoint = torch.load('models/exp/trained_exp_xtl_adam.pth')
    model_xtl_exp.load_state_dict(checkpoint)

    checkpoint = torch.load('models/exp/trained_exp_xsl_adam.pth')
    model_xsl_exp.load_state_dict(checkpoint)

    # checkpoint = torch.load('models/exp/trained_exp_agl_adam.pth')
    # model_agl_exp.load_state_dict(checkpoint)

    checkpoint = torch.load('models/exp/trained_exp_lcl_adam.pth')
    model_lcl_exp.load_state_dict(checkpoint)

    ds_loader = DataLoader(datasets.MNIST('datasets',
                                          train=False, download=True,
                                          transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                           batch_size=batch_size, shuffle=False)

    for ds in ds_loader:
        input_t = ds[0].to(DEVICE)
        input_t = input_t.view(input_t.size(0), -1)
        cntl_mse_output_t = model_cntl_mse.infer(input_t)
        cntl_mae_output_t = model_cntl_mae.infer(input_t)
        exp_xtl_output_t = model_xtl_exp.infer(input_t)
        exp_xsl_output_t = model_xsl_exp.infer(input_t)
        # exp_agl_output_t = model_agl_exp.infer(input_t)
        exp_lcl_output_t = model_lcl_exp.infer(input_t)
        truth_img = (0.5 * (input_t + 1)).clamp(0, 1).view(-1, 1, 28, 28)
        cntl_mse_mnist_reconstructed_img = (0.5 * (cntl_mse_output_t + 1)).clamp(0, 1).view(-1, 1, 28, 28)
        cntl_mae_mnist_reconstructed_img = (0.5 * (cntl_mae_output_t + 1)).clamp(0, 1).view(-1, 1, 28, 28)
        exp_xtl_mnist_reconstructed_img = (0.5 * (exp_xtl_output_t + 1)).clamp(0, 1).view(-1, 1, 28, 28)
        exp_xsl_mnist_reconstructed_img = (0.5 * (exp_xsl_output_t + 1)).clamp(0, 1).view(-1, 1, 28, 28)
        # exp_agl_mnist_reconstructed_img = (0.5 * (exp_agl_output_t + 1)).clamp(0, 1).view(-1, 1, 28, 28)
        exp_lcl_mnist_reconstructed_img = (0.5 * (exp_lcl_output_t + 1)).clamp(0, 1).view(-1, 1, 28, 28)
        save_image(truth_img, 'results/truth.png')
        save_image(cntl_mse_mnist_reconstructed_img, 'results/cntl/mnist_reconstructed_mse_adam.png')
        save_image(cntl_mae_mnist_reconstructed_img, 'results/cntl/mnist_reconstructed_mae_adam.png')
        save_image(exp_xtl_mnist_reconstructed_img, 'results/exp/mnist_reconstructed_xtl_adam.png')
        save_image(exp_xsl_mnist_reconstructed_img, 'results/exp/mnist_reconstructed_xsl_adam.png')
        # save_image(exp_agl_mnist_reconstructed_img, 'results/exp/mnist_reconstructed_agl_adam.png')
        save_image(exp_lcl_mnist_reconstructed_img, 'results/exp/mnist_reconstructed_lcl_adam.png')
        break

# ------------------------------------------------------------------------


if __name__ == '__main__':
    print('Exp1 MNIST Reconstruction.\n')

    run_training_adam(train_cntl=False,
                      train_exp_lcl=False,
                      train_exp_xtl=False,
                      train_exp_xsl=True,
                      train_exp_agl=False)
    run_cmp_prediction_adam()
