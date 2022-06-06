import torch
import torch.nn as nn

from Model import Model

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class ConvKB(Model):

    def __init__(self, config):
        super(ConvKB, self).__init__(config)

        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)

        self.conv1_bn = nn.BatchNorm2d(1)

        self.conv_layer1 = nn.Conv2d(1, self.config.out_channels, (1, 3))
        self.conv_layer2 = nn.Conv2d(1, self.config.out_channels, (3, 3))
        self.conv_layer3 = nn.Conv2d(1, self.config.out_channels, (5, 3))

        self.conv2_bn1 = nn.BatchNorm2d(self.config.out_channels)

        self.dropout = nn.Dropout(self.config.convkb_drop_prob)
        self.non_linearity = nn.ReLU()  # you should also tune with torch.tanh() or torch.nn.Tanh()

        num_layer = 11776

        self.fc_layer_alter = nn.Sequential(
            nn.Linear(num_layer, num_layer // 5),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(num_layer // 5, num_layer // 5),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(num_layer // 5, 1)
        )

        self.criterion = nn.Softplus()
        self.init_parameters()

    def init_parameters(self):
        if self.config.use_init_embeddings == False:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        else:
            self.ent_embeddings.weight.data = self.config.init_ent_embs
            self.rel_embeddings.weight.data = self.config.init_rel_embs

        nn.init.xavier_uniform_(self.fc_layer_alter[0].weight.data)
        nn.init.xavier_uniform_(self.fc_layer_alter[3].weight.data)
        nn.init.xavier_uniform_(self.fc_layer_alter[6].weight.data)
        nn.init.xavier_uniform_(self.conv_layer1.weight.data)
        nn.init.xavier_uniform_(self.conv_layer2.weight.data)
        nn.init.xavier_uniform_(self.conv_layer3.weight.data)

    def _calc(self, h, r, t):
        h = h.unsqueeze(1)  # bs x 1 x dim
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)

        conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim
        conv_input = conv_input.transpose(1, 2)
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        conv_input = self.conv1_bn(conv_input)

        batch_size = conv_input.shape[0]
        out_conv1 = self.conv_layer1(conv_input)
        out_conv2 = self.conv_layer2(conv_input)
        out_conv3 = self.conv_layer3(conv_input)

        out_conv = torch.cat([out_conv1, out_conv2, out_conv3], 2)
        out_conv = self.conv2_bn1(out_conv).view(batch_size, -1)
        out_conv = self.non_linearity(out_conv)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer_alter(input_fc).view(-1)

        return -score

    def loss(self, score, regul):
        return torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        r = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        score = self._calc(h, r, t)

        # regularization
        l2_reg = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        for W in self.conv_layer1.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.conv_layer2.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.conv_layer3.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.fc_layer_alter.parameters():
            l2_reg = l2_reg + W.norm(2)

        return self.loss(score, l2_reg)

    def predict(self):

        h = self.ent_embeddings(self.batch_h)
        r = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        score = self._calc(h, r, t)

        return score.cpu().data.numpy()
