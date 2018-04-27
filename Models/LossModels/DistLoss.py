import torch.nn


class DistLoss(torch.nn.Module):
    """
    Distance loss function.
    new loss function I thought of, it has no base ground i just thought its cool.
    given point x in R^n:

              1 - ||x|| if x <= inner_radius
    Loss(x) = 0 if inner_radius <= x <= outer_radius
              ||x|| - 2 if x >= outer_radius

    basically it means we allow the output to be between 2 balls of radius inner/outer radius
    """

    def __init__(self, inner_radius, outer_radius):
        super(DistLoss, self).__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def forward(self, point):
        return torch.mean(self.forward_vector(point))

    def forward_vector(self, point):
        l2norm = torch.pow(torch.pow(point, 2).sum(dim=1, keepdim=True), 0.5)
        loss = torch.abs(torch.clamp(l2norm, self.inner_radius, self.outer_radius) - l2norm)
        return loss
