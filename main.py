import os
import time

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader

from Models.DatasetModels.FasterDataset import SiameseNetworkDataset
from Models.LossModels.ContrastiveLoss import ContrastiveLoss
from Models.LossModels.DistLoss import DistLoss
from Models.NetworkModels.SiameseNet import SiameseNetwork
from utils.test_utils import eval_func, show_plot, test_net_by_hand, roc_curve, labels_and_predictions
from utils.test_utils import area_under_curve, move_data_to_variables_cuda, score_per_threshold, score_per_threshold2
from utils.save_and_load import save_checkpoint, load_checkpoint
import numpy as np

# settings
cuda = 3
transform = transforms.Compose([transforms.ToTensor()])
learning_rate = 0.0005
batch_size = 512
num_epochs = 20
train_net = True
num_images_to_plot = 1
net_path = 'checkpoint.pth.tar'
load_net = False
save_net = False
alpha_contrastive = 1.0
alpha_dist = 0.25  # parameter for combination of losses

# data sets
train_image_folder = dset.ImageFolder(root='data_35x35/train_set')
train_dataset = SiameseNetworkDataset(train_image_folder, transform=transform, should_invert=False, load_images=True, include_rotations=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
print("Training set length is {}".format(len(train_dataset)))

validation_image_folder = dset.ImageFolder(root='data_35x35/validation_set')
validation_dataset = SiameseNetworkDataset(validation_image_folder, transform=transform, should_invert=False, load_images=True, include_rotations=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=1000)
print("Validation set length is {}".format(len(validation_dataset)))

test_image_folder = dset.ImageFolder(root='data_35x35/Mnist_data')
test_dataset = SiameseNetworkDataset(test_image_folder, transform=transform, should_invert=False, load_images=True)
test_dataloader = DataLoader(test_dataset, batch_size=1)
print("Test set length is {}\n".format(len(test_dataset)))

# defining net, loss and optimizer
contrastive_loss = ContrastiveLoss()
dist_loss = DistLoss(0.5, 10)
net = SiameseNetwork(dropout=0.0, first_filter=4, output_size=10)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# check if saved file of model exists, if not continue without loading
if os.path.isfile(net_path) and load_net:
    load_checkpoint(net, optimizer, net_path)

if cuda >= 0:
    net.cuda(cuda)
    contrastive_loss.cuda(cuda)
    dist_loss.cuda(cuda)

if train_net:

    # variables to keep the progress
    loss_history = []
    auc_history = []
    counter = []
    auc_counter = []

    # training
    net.train()
    contrastive_loss.train()

    for epoch in range(num_epochs):
        if epoch == 10:
            alpha_dist = 0.0
        tic = time.clock()
        # train routine - maybe i want to define a function for it
        for data in train_dataloader:
            img0, img1, label = move_data_to_variables_cuda(data, cuda)

            output1, output2 = net(img0, img1)
            optimizer.zero_grad()
            loss = alpha_contrastive * contrastive_loss(output1, output2, label)\
                + alpha_dist * (dist_loss(output1) + dist_loss(output2))
            loss.backward()
            optimizer.step()

        # adding current loss on validation data
        print('epoch {} finished in {} seconds'.format(epoch, time.clock() - tic))
        # current_loss = eval_func(net, validation_dataloader, contrastive_loss, cuda)
        # loss_history.append(current_loss)
        # counter.append(epoch)
        # print('epoch {} loss is {}\n'.format(epoch, current_loss))

        # save progress
        if save_net:
            save_checkpoint(net, optimizer, net_path, cuda)

    # plot loss
    show_plot(counter, loss_history)

# find best threshold according to validation data
scores, thresholds = score_per_threshold(net, validation_dataloader, contrastive_loss, cuda)
best_score_arg = np.argmax(scores)
best_threshold = thresholds[best_score_arg]
print('The threshold is {}'.format(best_threshold))

# plt images and distances
test_net_by_hand(net, test_dataloader, num_images_to_plot, cuda)

# getting labels and predictions for dataset, plotting ROC and printing AUC ROC
test_dataloader = DataLoader(test_dataset, batch_size=1000)
labels, predictions = labels_and_predictions(net, test_dataloader, contrastive_loss, cuda)
labels = np.array(labels)
predictions = np.array(predictions)

# sorting by predictions
predictions_argsort = np.argsort(predictions)
labels = labels[predictions_argsort]
predictions = predictions[predictions_argsort]
class_predictions = predictions.copy()
class_predictions[predictions <= best_threshold] = 1
class_predictions[predictions > best_threshold] = 0
print('Accuracy on test data is {} for the picked threshold'.format(float(np.count_nonzero(class_predictions == labels)) / len(labels)))
scores, thresholds = score_per_threshold2(labels, predictions)
show_plot(thresholds, scores)
best_score_arg = np.argmax(scores)
print('Accuracy on test data is {} for the best threshold ({})'.format(scores[best_score_arg], thresholds[best_score_arg]))

x_axis, y_axis = roc_curve(labels, predictions)
show_plot(x_axis, y_axis)
print("area under curve is {}".format(area_under_curve(x_axis, y_axis)))

