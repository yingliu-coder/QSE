import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

SEQ_LEN = 48
channel = 35
num = 19

def quantile_loss(preds, target, quantiles, mask):
    loss = 0
    for i, q in enumerate(quantiles):
        forecast = preds[:, i, :]
        loss += torch.sum(
            torch.abs((forecast - target) * mask * ((target <= forecast) * 1.0 - q))
        )
    loss = loss / len(quantiles)
    return loss


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size) 
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model(nn.Module):
    def __init__(self, rnn_hid_size):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(channel * 2, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size=channel, output_size= self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size=channel, output_size=channel, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, channel)
        self.feat_reg = FeatureRegression(channel)

        # self.weight_combine = nn.Linear(35 * 2, 35)

        self.weight_combine_linears = nn.ModuleList([nn.Linear(channel * 2, channel) for i in range(num)])

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)


    def forward(self, x, mask, delta):
        
        values = x
        masks = mask
        deltas = delta

        
        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0

        imputations = []
        ensembles = []

        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            loss_1 = torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)
            x_loss += loss_1
            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            loss_2 = torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)
            x_loss += loss_2

            ensemble_step_list = []
            m_list = []
            x_list = []
            for i in range(num):
                ensembles_alpha = self.weight_combine_linears[i](torch.cat([gamma_x, m], dim=1))
                temp_h = ensembles_alpha * z_h + (1 - ensembles_alpha) * x_h
                m_list.append(m.unsqueeze(dim=1))
                x_list.append(x.unsqueeze(dim=1))
                ensemble_step_list.append(temp_h.unsqueeze(dim=1))

            ensemble_step = torch.cat(ensemble_step_list, dim=1)
            ensemble_m = torch.cat(m_list, dim=1)
            ensemble_x = torch.cat(x_list, dim=1)
            # quantiles = np.arange(0.05, 1.0, 0.05)
            quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
            loss_3 = quantile_loss(ensemble_step, x, quantiles, m) / (torch.sum(m) + 1e-5)
            x_loss += loss_3
            ensemble_step_mean = torch.mean(ensemble_step, dim=1)
            x_loss += torch.sum(torch.abs(x - ensemble_step_mean) * m) / (torch.sum(m) + 1e-5)
            ensemble_h = ensemble_m * ensemble_x + (1 - ensemble_m) * ensemble_step

            c_c = torch.mean(ensemble_h, dim=1)

            inputs = torch.cat([c_c, m], dim = 1)
            h, c = self.rnn_cell(inputs, (h, c))
            imputations.append(c_c.unsqueeze(dim=1))
            ensembles.append(ensemble_h.unsqueeze(dim=2))

        x_loss = x_loss / SEQ_LEN
        imputations = torch.cat(imputations, dim=1)
        ensembles = torch.cat(ensembles, dim=2)
        return {'loss': x_loss, 'imputations': imputations, 'ensembles': ensembles}

