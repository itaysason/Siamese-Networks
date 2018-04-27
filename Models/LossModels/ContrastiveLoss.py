import torch.nn
from torch.nn.functional import pairwise_distance


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        return torch.mean(self.forward_vector(output1, output2, label))

    def forward_vector(self, output1, output2, label):
        euclidean_distance = pairwise_distance(output1, output2)
        loss_contrastive_vec = (1-label) * torch.pow(euclidean_distance, 2) +\
                           label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0, max=self.margin), 2)

        return loss_contrastive_vec
