import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.autograd import Variable
from torch.nn.functional import pairwise_distance
import torch


def show_image_text(img, text=None):
    np_img = img.numpy()
    plt.axis("off")
    if text is not None:
        plt.text(0, 0, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def show_plot(x_axis, y_axis):
    plt.plot(x_axis, y_axis)
    plt.show()


def eval_func(net, dataloader, loss_function, cuda):

    # check net and loss training status to return it to the way it was
    net_training_status = False
    if net.training:
        net_training_status = True
        net.eval()

    loss_training_status = False
    if net.training:
        loss_training_status = True
        net.eval()

    total_loss = 0
    for data in dataloader:
        img0, img1, label = move_data_to_variables_cuda(data, cuda)

        output1, output2 = net(img0, img1)
        total_loss += loss_function(output1, output2, label).data[0] * img0.shape[0]

    # return to the original status
    if net_training_status:
        net.train()
    if loss_training_status:
        net.train()

    return total_loss


def test_net_by_hand(net, dataloader, num_images, cuda):

    # check network training status to return it to the way it was
    training_status = False
    if net.training:
        training_status = True
        net.eval()

    for i, data in enumerate(dataloader):
        if i >= num_images:
            break
        img0, img1, label = data
        concatenated_images = torch.cat((img0, img1), 0)

        img0, img1, label = move_data_to_variables_cuda(data, cuda)

        output1, output2 = net(img0, img1)
        euclidean_distance = pairwise_distance(output1, output2)
        if int(label):
            s = 'different'
        else:
            s = 'same'
        text = 'Dissimilarity: {:.2f}, {}'.format(float(euclidean_distance.data), s)
        im = torchvision.utils.make_grid(concatenated_images)
        show_image_text(im.cpu(), text)

    # return to the original status
    if training_status:
        net.train()


def labels_and_predictions(net, dataloader, loss_function, cuda):

    # check net and loss training status to return it to the way it was
    net_training_status = False
    if net.training:
        net_training_status = True
        net.eval()
    loss_training_status = False
    if net.training:
        loss_training_status = True
        net.eval()

    labels = []
    predictions = []
    for i, data in enumerate(dataloader):
        img0, img1, label = move_data_to_variables_cuda(data, cuda)

        output1, output2 = net(img0, img1)
        loss_vector = loss_function.forward_vector(output1, output2, label)

        labels.extend(label.cpu().data.numpy().transpose()[0].tolist())
        predictions.extend(loss_vector.cpu().data.numpy().transpose()[0].tolist())

    # return to the original status
    if net_training_status:
        net.train()
    if loss_training_status:
        net.train()

    return labels, predictions


def score_per_threshold2(labels, predictions):
    labels = np.array(labels)
    predictions = np.array(predictions)

    # sorting by predictions
    predictions_argsort = np.argsort(predictions)
    labels = labels[predictions_argsort]
    predictions = predictions[predictions_argsort]

    # thresholds are just the unique predictions, and for each threshold we compute number of true predictions
    thresholds = np.unique(predictions)
    num_true = np.zeros(thresholds.shape)
    labels_reverse = 1 - labels
    num_true[0] = np.sum(labels_reverse[1:]) + labels[0]
    ind = 0
    i = 0
    while i < len(predictions) - 1:
        if predictions[i] == predictions[i + 1]:
            i += 1
            num_true[ind] += labels[i] - labels_reverse[i]
        else:
            i += 1
            num_true[ind + 1] = num_true[ind] + labels[i] - labels_reverse[i]
            ind += 1

    test_score = num_true / len(labels)
    return test_score, thresholds


def roc_curve(labels, predictions):

    labels = np.array(labels)
    predictions = np.array(predictions)/np.max(predictions)

    # sorting by predictions
    predictions_argsort = np.argsort(predictions)
    labels = labels[predictions_argsort]
    predictions = predictions[predictions_argsort]

    # computing true/false positives rates for each threshold and assign output arrays
    x_axis = [0]
    y_axis = [0]
    true_positives = 0
    false_positive = 0
    total_positives = np.sum(labels)
    total_negatives = len(labels) - total_positives
    for i in range(len(predictions)):
        # if label is positive update true_positives and y_axis
        if labels[i] == 1:
            true_positives += 1
            y_axis[-1] = true_positives / total_positives
        # else x_axis changed and we start a new point
        else:
            false_positive += 1
            x_axis.append(false_positive / total_negatives)
            y_axis.append(true_positives / total_positives)

    return x_axis, y_axis


def score_per_threshold(net, dataloader, loss_function, cuda):
    """
    Evaluating the score True/Total for every possible threshold over the dataloader. For threshold t if the
    prediction <= t we accept, else we reject
    :param net:
    :param dataloader:
    :param loss_function:
    :param cuda:
    :return:
    """

    # check net and loss training status to return it to the way it was
    net_training_status = False
    if net.training:
        net_training_status = True
        net.eval()
    loss_training_status = False
    if net.training:
        loss_training_status = True
        net.eval()

    labels = []
    predictions = []
    for i, data in enumerate(dataloader):
        img0, img1, label = move_data_to_variables_cuda(data, cuda)

        output1, output2 = net(img0, img1)
        loss_vector = loss_function.forward_vector(output1, output2, label)

        labels.extend(label.cpu().data.numpy().transpose()[0].tolist())
        predictions.extend(loss_vector.cpu().data.numpy().transpose()[0].tolist())

    # return to the original status
    if net_training_status:
        net.train()
    if loss_training_status:
        net.train()

    labels = np.array(labels)
    predictions = np.array(predictions)

    # sorting by predictions
    predictions_argsort = np.argsort(predictions)
    labels = labels[predictions_argsort]
    predictions = predictions[predictions_argsort]

    # thresholds are just the unique predictions, and for each threshold we compute number of true predictions
    thresholds = np.unique(predictions)
    num_true = np.zeros(thresholds.shape)
    labels_reverse = 1 - labels
    num_true[0] = np.sum(labels_reverse[1:]) + labels[0]
    ind = 0
    i = 0
    while i < len(predictions) - 1:
        if predictions[i] == predictions[i + 1]:
            i += 1
            num_true[ind] += labels[i] - labels_reverse[i]
        else:
            i += 1
            num_true[ind + 1] = num_true[ind] + labels[i] - labels_reverse[i]
            ind += 1

    test_score = num_true / len(labels)

    return test_score, thresholds


def area_under_curve(x_axis, y_axis):
    auc = 0
    for i in range(len(x_axis) - 1):
        # trapezoid area
        auc += (x_axis[i + 1] - x_axis[i]) * (y_axis[i + 1] + y_axis[i]) / 2
    return auc


def move_data_to_variables_cuda(data, cuda):
    if cuda < 0:
        output = [Variable(var) for var in data]
    else:
        output = [Variable(var).cuda(cuda) for var in data]
    return tuple(output)
