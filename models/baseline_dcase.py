#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Module
from torch import Tensor
import torch
from modules import Encoder
from modules import Decoder

__author__ = 'Wu Qianyang, Tao Shengqi, Yang Xingyu'
__docformat__ = 'reStructuredText'
__all__ = ['BaselineDCASE']


class BaselineDCASE(Module):

    def __init__(self,
                 in_channels: int,
                 t_steps: int,
                 input_dim_encoder: int,
                 output_dim_encoder: int,
                 output_dim_h_decoder: int,
                 nb_classes: int,
                 dropout_p_decoder: float,
                 max_out_t_steps: int) \
            -> None:
        """Baseline method for audio captioning with Clotho dataset.

        :param in_channels: Input channels of CNN
        :type in_channels: int
        :param t_steps: Normalized length of input data at time dimension
        :type t_steps: int
        :param input_dim_encoder: Input dimensionality of the full connect layer of the encoder
        :type input_dim_encoder: int
        :param output_dim_encoder: Output dimensionality of the full connect layer of the encoder
        :type output_dim_encoder: int
        :param output_dim_h_decoder: Hidden output dimensionality of the decoder.
        :type output_dim_h_decoder: int
        :param nb_classes: Amount of output classes.
        :type nb_classes: int
        :param dropout_p_decoder: Decoder RNN dropout.
        :type dropout_p_decoder: float
        :param max_out_t_steps: Maximum output time-steps of the decoder.
        :type max_out_t_steps: int
        """
        super().__init__()
        self.in_channels: int = in_channels
        self.t_steps: int = t_steps
        self.max_out_t_steps: int = max_out_t_steps
        self.encoder:Module = Encoder(
            in_dim = input_dim_encoder,
            out_dim = output_dim_encoder)
        self.decoder: Module = Decoder(
            input_dim=1,
            output_dim=output_dim_h_decoder,
            nb_classes=nb_classes,
            dropout_p=dropout_p_decoder)

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the baseline method.

        :param x: Input features.
        :type x: torch.Tensor
        :return: Predicted values.
        :rtype: torch.Tensor
        """
        #padding
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        padd_dim = self.t_steps - x.shape[1];
        x0 = torch.zeros(x.shape[0], padd_dim, x.shape[2]);
        x0 = x0.to(device);
        x = torch.cat((x, x0), 1);
        del x0;
        x = x.unsqueeze(1).expand(-1, self.in_channels, -1, -1) 

        h_encoder = self.encoder(x);
        h_encoder = h_encoder.unsqueeze(2);
        return self.decoder(h_encoder)


# EOF
