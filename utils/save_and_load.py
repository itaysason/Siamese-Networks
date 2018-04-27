import torch


def save_checkpoint(net, optimizer, filename='checkpoint.pth.tar', cuda=False):

    if cuda:
        net.cpu()

    torch.save({
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filename)

    if cuda:
        net.cuda()


def load_checkpoint(net, optimizer, net_path='checkpoint.pth.tar'):

    checkpoint = torch.load(net_path)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("checkpoint loaded")
