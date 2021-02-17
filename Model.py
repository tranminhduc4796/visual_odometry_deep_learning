# File to return the Deep VO model.
from collections import OrderedDict

import torch.nn.functional as F
import torch
from torch import nn


# DeepVO model
class DeepVO(nn.Module):
    def __init__(self, img_w, img_h, seq_len, batch_size, activation='relu', parameterization='default',
                 dropout=0.3, flownet_weights_path=None, num_lstm_cells=2):
        super(DeepVO, self).__init__()

        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)

        if self.img_w < 64 or self.img_h < 64:
            raise ValueError('The width and height for an input image must be at least 64 px.')

        # The feature map's width and height are scaled down 64 times after Conv.
        self.lstm_input_size = int(1024 * (self.img_w * self.img_h) / (64 * 64))

        self.activation = activation
        self.parameterization = parameterization
        if parameterization == 'quaternion':
            self.rotationDims = 4
        else:
            self.rotationDims = 3
        self.translationDims = 3

        if dropout <= 0.0:
            self.dropout = False
        else:
            self.dropout = True
            self.drop_ratio = dropout

        self.num_lstm_cells = num_lstm_cells
        # Path to FlowNet weights
        if flownet_weights_path is not None and flownet_weights_path != '':
            self.use_flownet = True
            self.flownet_weights_path = flownet_weights_path
            self.load_flownet_weights()

        # CONV Architecture
        self.flownet = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=True)),
                ('l_relu1', nn.LeakyReLU()),
                ('conv2', nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=True)),
                ('l_relu2', nn.LeakyReLU()),
                ('conv3', nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=True)),
                ('l_relu3', nn.LeakyReLU()),
                ('conv3_1', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)),
                ('l_relu3_1', nn.LeakyReLU()),
                ('conv4', nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True)),
                ('l_relu4', nn.LeakyReLU()),
                ('conv4_1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)),
                ('l_relu4_1', nn.LeakyReLU()),
                ('conv5', nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=True)),
                ('l_relu5', nn.LeakyReLU()),
                ('conv5_1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)),
                ('l_relu5_1', nn.LeakyReLU()),
                ('conv6', nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=True))
            ])
        )

        # LSTM for rotation and translations
        self.LSTM_R = nn.LSTM(self.lstm_input_size, 1024, self.num_lstm_cells)
        self.LSTM_T = nn.LSTM(self.lstm_input_size, 1024, self.num_lstm_cells)

        self.fc1_R = nn.Linear(1024, 128)
        self.fc1_T = nn.Linear(1024, 128)

        self.fc2_R = nn.Linear(128, 32)
        self.fc2_T = nn.Linear(128, 32)

        if self.parameterization == 'quaternion':
            self.fc_rot = nn.Linear(32, 4)
        else:
            self.fc_rot = nn.Linear(32, 3)

        self.fc_trans = nn.Linear(32, 3)

    def forward(self, input_tensor):
        """
        :param input_tensor: Shape of [Batch size x Sequence Length x Channel x Width x Height]
        """
        feature_maps = []

        # Forward sequences of images into flownet one by one
        for x in input_tensor:
            feature_map = self.flownet(x)
            feature_maps.append(feature_map)

        input_tensor = torch.stack(feature_maps)

        # Reshape output of Conv to feed into LSTM
        input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        input_tensor = input_tensor.view(self.seq_len, self.batch_size, self.lstm_input_size)

        o_R, (h_R, c_R) = self.LSTM_R(input_tensor)
        o_T, (h_T, c_T) = self.LSTM_T(input_tensor)

        # Forward pass through the FC layers
        output_fc1_R = F.relu(self.fc1_R(o_R))
        output_fc1_T = F.relu(self.fc1_T(o_T))

        if self.activation == 'selu':
            output_fc1_R = F.selu(self.fc1_R(o_R))
            output_fc1_T = F.selu(self.fc1_T(o_T))

        output_fc2_R = self.fc2_R(output_fc1_R)
        output_fc2_T = self.fc2_T(output_fc1_T)

        if self.dropout is True:
            output_fc2_R = F.dropout(self.fc2_R(output_fc1_R), p=self.drop_ratio, training=self.training)
            output_fc2_T = F.dropout(self.fc2_T(output_fc1_T), p=self.drop_ratio, training=self.training)

        output_rot = self.fc_rot(output_fc2_R)
        output_trans = self.fc_trans(output_fc2_T)
        return output_rot, output_trans

    # Initialize the weights of the network
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        # Forget gate bias trick: Initially during training, it is often helpful
                        # to initialize the forget gate bias to a large value, to help information
                        # flow over longer time steps.
                        # In a PyTorch LSTM, the biases are stored in the following order:
                        # [ b_ig | b_fg | b_gg | b_og ]
                        # where, b_ig is the bias for the input gate,
                        # b_fg is the bias for the forget gate,
                        # b_gg (see LSTM docs, Variables section),
                        # b_og is the bias for the output gate.
                        # So, we compute the location of the forget gate bias terms as the
                        # middle one-fourth of the bias vector, and initialize them.

                        # First initialize all biases to zero
                        nn.init.constant_(param, 0.)
                        bias = getattr(m, name)
                        n = bias.size(0)
                        start, end = n // 4, n // 2
                        bias.data[start:end].fill_(1.)

    def load_flownet_weights(self):
        pretrained_weights = torch.load(self.flownet_weights_path)['state_dict']
        update_dict = {param: val for param, val in
                       zip(self.flownet.state_dict().keys(), pretrained_weights['state_dict'].values())}
        self.flownet.load_state_dict(update_dict)
        # Freeze pre-trained weights
        for name, param in self.flownet.named_parameters():
            param.requires_grad = False
