import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, dropout=0.05, first_filter=4, output_size=10):

        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(

            nn.Conv2d(1, first_filter, kernel_size=8, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(first_filter),
            nn.Dropout2d(p=dropout, inplace=True),

            nn.Conv2d(first_filter, 2*first_filter, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*first_filter),
            nn.Dropout2d(p=dropout, inplace=True),

            nn.Conv2d(2*first_filter, 2*2*first_filter, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*2*first_filter),
            nn.Dropout2d(p=dropout, inplace=True),

            nn.Conv2d(2*2*first_filter, 2*2*2*first_filter, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*2*2*first_filter),
            nn.Dropout2d(p=dropout, inplace=True),

        )

        # 19x19 is only for image size 105, for other sizes need to compute
        self.fc1 = nn.Sequential(
            nn.Linear(2*2*2*first_filter * 19 * 19, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, output_size))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
