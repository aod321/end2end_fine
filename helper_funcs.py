import torch
import numpy as np
import matplotlib.pyplot as plt


class F1Score(torch.nn.CrossEntropyLoss):
    def __init__(self, device):
        super(F1Score, self).__init__()
        self.device = device
        self.name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
        self.F1_name_list = ['eyebrows', 'eyes', 'nose', 'u_lip', 'i_mouth', 'l_lip', 'mouth_all']

        self.TP = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.FP = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.TN = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.FN = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.recall = {x: 0.0 + 1e-20
                       for x in self.F1_name_list}
        self.precision = {x: 0.0 + 1e-20
                          for x in self.F1_name_list}
        self.F1_list = {x: []
                        for x in self.F1_name_list}
        self.F1 = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}

        self.recall_overall_list = {x: []
                                    for x in self.F1_name_list}
        self.precision_overall_list = {x: []
                                       for x in self.F1_name_list}
        self.recall_overall = 0.0
        self.precision_overall = 0.0
        self.F1_overall = 0.0

    def forward(self, predict, labels):
        part_name_list = {1: 'eyebrow1', 2: 'eyebrow2', 3: 'eye1', 4: 'eye2',
                          5: 'nose', 6: 'u_lip', 7: 'i_mouth', 8: 'l_lip'}
        F1_name_list_parts = ['eyebrow1', 'eyebrow2',
                              'eye1', 'eye2',
                              'nose', 'u_lip', 'i_mouth', 'l_lip']
        TP = {x: 0.0 + 1e-20
              for x in F1_name_list_parts}
        FP = {x: 0.0 + 1e-20
              for x in F1_name_list_parts}
        TN = {x: 0.0 + 1e-20
              for x in F1_name_list_parts}
        FN = {x: 0.0 + 1e-20
              for x in F1_name_list_parts}
        pred = predict.argmax(dim=1, keepdim=False)
        # ground = labels.argmax(dim=1, keepdim=False)
        ground = labels.long()
        assert ground.shape == pred.shape
        for i in range(1, 9):
            TP[part_name_list[i]] += ((pred == i) * (ground == i)).sum().tolist()
            TN[part_name_list[i]] += ((pred != i) * (ground != i)).sum().tolist()
            FP[part_name_list[i]] += ((pred == i) * (ground != i)).sum().tolist()
            FN[part_name_list[i]] += ((pred != i) * (ground == i)).sum().tolist()

        for r in ['eyebrow1', 'eyebrow2']:
            self.TP['eyebrows'] += TP[r]
            self.TN['eyebrows'] += TN[r]
            self.FP['eyebrows'] += FP[r]
            self.FN['eyebrows'] += FN[r]

        for r in ['eye1', 'eye2']:
            self.TP['eyes'] += TP[r]
            self.TN['eyes'] += TN[r]
            self.FP['eyes'] += FP[r]
            self.FN['eyes'] += FN[r]

        for r in ['u_lip', 'i_mouth', 'l_lip']:
            self.TP[r] += TP[r]
            self.TN[r] += TN[r]
            self.FP[r] += FP[r]
            self.FN[r] += FN[r]
            self.TP['mouth_all'] += TP[r]
            self.TN['mouth_all'] += TN[r]
            self.FP['mouth_all'] += FP[r]
            self.FN['mouth_all'] += FN[r]

        for r in ['nose']:
            self.TP[r] += TP[r]
            self.TN[r] += TN[r]
            self.FP[r] += FP[r]
            self.FN[r] += FN[r]

        for r in self.F1_name_list:
            self.recall[r] = self.TP[r] / (
                    self.TP[r] + self.FP[r])
            self.precision[r] = self.TP[r] / (
                    self.TP[r] + self.FN[r])
            self.recall_overall_list[r].append(self.recall[r])
            self.precision_overall_list[r].append(self.precision[r])
            self.F1_list[r].append((2 * self.precision[r] * self.recall[r]) /
                                   (self.precision[r] + self.recall[r]))
        return self.F1_list, self.recall_overall_list, self.precision_overall_list

    def output_f1_score(self):
        # print("All F1_scores:")
        for x in self.F1_name_list:
            self.recall_overall_list[x] = np.array(self.recall_overall_list[x]).mean()
            self.precision_overall_list[x] = np.array(self.precision_overall_list[x]).mean()
            self.F1[x] = np.array(self.F1_list[x]).mean()
            print("{}:{}\t".format(x, self.F1[x]))
        for x in self.F1_name_list:
            self.recall_overall += self.recall_overall_list[x]
            self.precision_overall += self.precision_overall_list[x]
        self.recall_overall /= len(self.F1_name_list)
        self.precision_overall /= len(self.F1_name_list)
        self.F1_overall = (2 * self.precision_overall * self.recall_overall) / \
                          (self.precision_overall + self.recall_overall)
        # print("{}:{}\t".format("overall", self.F1_overall))
        return self.F1, self.F1_overall


def imshow_func(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def calc_centroid(tensor):
    # Inputs Shape(N, 9 , 64, 64)
    # Return Shape(N, 9 ,2)
    input = tensor.float() + 1e-10
    n, l, h, w = input.shape
    indexs_y = torch.from_numpy(np.arange(h)).float().to(tensor.device)
    indexs_x = torch.from_numpy(np.arange(w)).float().to(tensor.device)
    center_y = input.sum(3) * indexs_y.view(1, 1, -1)
    center_y = center_y.sum(2, keepdim=True) / input.sum([2, 3]).view(n, l, 1)
    center_x = input.sum(2) * indexs_x.view(1, 1, -1)
    center_x = center_x.sum(2, keepdim=True) / input.sum([2, 3]).view(n, l, 1)
    output = torch.cat([center_y, center_x], 2)
    # output = torch.cat([center_x, center_y], 2)
    return output
