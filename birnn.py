import torch
import torch.nn as nn
from torch.autograd import Variable


import rnn


SEQ_LEN = 48
RNN_HID_SIZE = 108


class Model(nn.Module):
    def __init__(self, rnn_hid_size):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size

        self.build()

    def build(self):
        self.rnn_f = rnn.Model(self.rnn_hid_size)
        self.rnn_b = rnn.Model(self.rnn_hid_size)

    def forward(self, x_f, masks_f, deltas_f, x_b, masks_b, deltas_b):
        ret_f = self.rnn_f(x_f, masks_f, deltas_f)
        ret_b = self.reverse(self.rnn_b(x_b, masks_b, deltas_b))

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])
        loss = loss_f + loss_b + loss_c
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2
        ensembles = (ret_f['ensembles'] + ret_b['ensembles']) / 2
        ret_f['loss'] = loss
        ret_f['imputations'] = imputations
        ret_f['ensembles'] = ensembles

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_, pos):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[pos])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad = False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(pos, indices)

        for key in ret:
            if key != 'ensembles':
                ret[key] = reverse_tensor(ret[key], 1)
            else:
                ret[key] = reverse_tensor(ret[key], 2)

        return ret

