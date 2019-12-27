from icnnmodel import FaceModel, Stage2FaceModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Stage1Model(pl.LightningModule):
    def __init__(self):
        super(Stage1Model, self).__init__()
        self.model = FaceModel()

    def forward(self, x):
        y = self.model.forward(x)
        return y


class Stage2Model(pl.LightningModule):
    def __init__(self):
        super(Stage2Model, self).__init__()
        self.model = nn.ModuleList([Stage2FaceModel()
                                    for _ in range(4)])
        for i in range(3):
            self.model[i].set_label_channels(2)
        self.model[3].set_label_channels(4)

    def forward(self, parts):
        eyebrow1_pred = self.model[0](parts[:, 0])
        eyebrow2_pred = torch.flip(self.model[0](torch.flip(parts[:, 1], [3])), [3])
        eye1_pred = self.model[1](parts[:, 2])
        eye2_pred = torch.flip(self.model[1](torch.flip(parts[:, 3], [3])), [3])
        nose_pred = self.model[2](parts[:, 4])
        mouth_pred = self.model[3](parts[:, 5])
        predict = [eyebrow1_pred, eyebrow2_pred,
                   eye1_pred, eye2_pred, nose_pred, mouth_pred]
        for i in range(len(predict)):
            predict[i] = F.softmax(predict[i], dim=1)

        return predict


class SelectNet(nn.Module):
    def __init__(self):
        super(SelectNet, self).__init__()
        self.theta = None
        self.points = None
        self.device = None
        self.img = None
        self.label = None

    def forward(self, img, label, points):
        self.points = points
        self.img = img
        self.label = label
        self.device = img.device
        n, l, h, w = img.shape
        self.points = torch.cat([self.points[:, 1:6],
                                 self.points[:, 6:9].mean(dim=1, keepdim=True)],
                                dim=1)
        assert self.points.shape == (n, 6, 2)
        self.theta = torch.zeros((n, 6, 2, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        # I'm going to cut all of parts into 81x81
        # Eyes,eyebrows,nose theta
        for i in range(5):
            self.theta[:, i, 0, 0] = (81 -1) / (w -1)
            self.theta[:, i, 0, 2] = -1 + (2 * self.points[:, i, 1]) / (w -1)
            self.theta[:, i, 1, 1] = (81 -1) / h
            self.theta[:, i, 1, 2] = -1 + (2 * self.points[:, i, 0]) / (h -1)
        # Mouth theta
        for i in range(5, 6):
            self.theta[:, i, 0, 0] = (81-1) / (w-1)
            self.theta[:, i, 0, 2] = -1 + (2 * self.points[:, i, 1]) / (w - 1)
            self.theta[:, i, 1, 1] = (81-1) / h
            self.theta[:, i, 1, 2] = -1 + (2 * self.points[:, i, 0]) / (h - 1)
        croped_image = self.crop_image()
        croped_pred = self.crop_label()
        return croped_image, croped_pred

    def crop_image(self):
        N = self.img.shape[0]
        img_in = self.img
        theta_in = self.theta
        samples = []
        for i in range(6):
            grid = F.affine_grid(theta_in[:, i], [N, 3, 81, 81], align_corners=True).to(self.device)
            samples.append(F.grid_sample(input=img_in, grid=grid, align_corners=True,
                                         mode='bilinear', padding_mode='zeros'))
        samples = torch.stack(samples, dim=1)
        assert samples.shape == (N, 6, 3, 81, 81)
        return samples

    def crop_label(self):
        labels = []
        theta = self.theta
        label = self.label.float()
        N, L = label.shape[0], label.shape[1]
        for i in range(6):
            grid = F.affine_grid(theta[:, i], [N, L, 81, 81], align_corners=True).to(label.device)
            labels.append(
                F.grid_sample(input=torch.unsqueeze(label, dim=1), mode='nearest', grid=grid, align_corners=True))
        labels = torch.cat(labels, dim=1)
        assert labels.shape == (N, 6, 81, 81)
        mouth_labels = labels[:, 5:6]
        mouth_labels[(mouth_labels != 6) * (mouth_labels != 7) * (mouth_labels != 8)] = 0  # bg
        mouth_labels[mouth_labels == 6] = 1  # up_lip
        mouth_labels[mouth_labels == 7] = 2  # in_mouth
        mouth_labels[mouth_labels == 8] = 3  # down_lip
        for i in range(5):
            labels[:, i][labels[:, i] != i + 1] = 0
            labels[:, i][labels[:, i] == i + 1] = 1

        labels = torch.cat([labels[:, 0:5], mouth_labels], dim=1)
        assert labels.shape == (N, 6, 81, 81)
        return labels


class ReverseNet(nn.Module):
    def __init__(self):
        super(ReverseNet, self).__init__()
        self.device = None
        self.rtheta = None
        self.preds = None
        self.rtheta = None

    def forward(self, preds, theta):
        self.preds = preds
        self.theta = theta
        self.device = theta.device
        N = theta.shape[0]

        ones = torch.tensor([[0., 0., 1.]]).repeat(N, 6, 1, 1).to(self.device)

        self.rtheta = torch.cat([self.theta, ones], dim=2).to(self.device)
        self.rtheta = torch.inverse(self.rtheta)
        self.rtheta = self.rtheta[:, :, 0:2]
        assert self.rtheta.shape == (N, 6, 2, 3)
        del ones

        reversed_pred = self.reverse_predict()
        # Output Pred(N, 9, 512, 512)

        return reversed_pred

    def reverse_predict(self):
        fg = []
        bg = []
        N = self.rtheta.shape[0]
        for i in range(6):
            all_pred = F.softmax(self.preds[i], dim=1)
            grid = F.affine_grid(theta=self.rtheta[:, i], size=[N, all_pred.shape[1], 512, 512], align_corners=True).to(
                self.device)
            bg_grid = F.affine_grid(theta=self.rtheta[:, i], size=[N, 1, 512, 512], align_corners=True).to(
                self.device)
            temp = F.grid_sample(input=all_pred, grid=grid, mode='nearest', padding_mode='zeros', align_corners=True)
            temp2 = F.grid_sample(input=all_pred[:, 0:1], grid=bg_grid, mode='nearest', padding_mode='border',
                                  align_corners=True)
            bg.append(temp2)
            fg.append(temp[:, 1:])
            del temp, temp2
        bg = torch.cat(bg, dim=1)
        bg = (bg[:, 0:1] * bg[:, 1:2] * bg[:, 2:3] * bg[:, 3:4] *
              bg[:, 4:5] * bg[:, 5:6])
        fg = torch.cat(fg, dim=1)  # Shape(N, 8, 512 ,512)
        sample = torch.cat([bg, fg], dim=1)
        assert sample.shape == (N, 9, 512, 512)
        return sample
