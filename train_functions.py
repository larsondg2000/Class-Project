"""
Performs all the model training functions:
    1) Gets arg parse values
    2) Loads datasets and transforms
    3) Trains model using VGG19 or Alexnet
    3) Saves model as a checkpoint
    4) User can set epochs, gpu/cpu, learn rate, hidden layers, save and data dir
    5) User can select model VGG19 or AlexNet
"""

# Imports here
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse

"""
    Arg Parse Parameters
    - save_dir: is where the model is stored after training
    - arch: model type (vgg19 or AlexNet)
    - learning_rate: training learning rate
    - epochs: number of epochs for training
    - gpu: set to gpu (cuda) or cpu
"""
def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", action="store", dest="save_dir",
                        default="/home/workspace/ImageClassifier/checkpoint.pth")
    parser.add_argument('--data_dir', type=str, dest='data_dir', action="store", nargs="*", default="flowers")
    parser.add_argument('--model', dest='model', default='vgg19', choices=['alexnet', 'vgg19'])
    parser.add_argument('--hidden_layers', type=int, dest='hidden_layers', default='4096')
    parser.add_argument('--learn_rate', type=float, dest='learn_rate', default='0.0001')
    parser.add_argument('--epochs', type=int, dest='epochs', default='1')
    parser.add_argument('--gpu', action="store_true", default=True)
    return parser.parse_args()


"""
Train Function
    -sets up data transformer and loaders
    - calls model to be trained (VGG18 and AlexNet)
"""


def train(model, epochs, gpu, learn_rate, hidden_layers, save_dir, data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {

        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),

        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

        ]),

        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # set batch size so easy to adjust later
    b_size = 64

    train_loader = DataLoader(image_datasets['train'], batch_size=b_size, shuffle=True)
    valid_loader = DataLoader(image_datasets['valid'], batch_size=b_size)
    test_loader = DataLoader(image_datasets['test'], batch_size=b_size)

    # Call Training Model
    if model == 'alexnet':
        model_alexnet(epochs, learn_rate, train_loader,
                      test_loader, image_datasets, gpu, hidden_layers, save_dir)
    if model == 'vgg19':
        model_vgg(epochs, learn_rate, train_loader,
                  test_loader, image_datasets, gpu, hidden_layers, save_dir)

# VGG 19 Model-saves model at end of training


def model_vgg(epochs, learn_rate, train_loader, test_loader, image_datasets, gpu, hidden_layers, save_dir):
    model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # check the model set up
    print("********* Model Setup Inputs***********")
    print("Epochs = {}  Learn Rate = {}  Hidden Layers = {} Model = VGG19".format(epochs, learn_rate, hidden_layers))
    print("")
    print("**** These are the model settings for VGG19 before updating the classifier ****")
    print(model)

    # Setup classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_layers)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.4)),
        ('output', nn.Linear(hidden_layers, 102)),
        ('dropout', nn.Dropout(p=0.4)),
        ('softmax', nn.LogSoftmax(dim=1))
    ]))

    # Replace classifier
    model.classifier = classifier

    # send to cuda or cpu
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()

    # Add loss calc and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    # inits
    steps = 0
    running_loss = 0
    print_every = 50

    print("********** Learn Rate= {} *********".format(learn_rate))

    # Training steps
    for epoch in range(epochs):

        print("Epoch: {}/{}".format(epoch + 1, epochs))

        for inputs, labels in train_loader:
            steps += 1

            # Move input and label tensors to the default device
            if gpu and cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs = inputs.cpu()
                labels = labels.cpu()

            # zero grads
            optimizer.zero_grad()

            # forward
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            # back
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        if gpu and cuda:
                            inputs = inputs.cuda()
                            labels = labels.cuda()

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.sum(equals.type(torch.FloatTensor)).item()

                train_loss = running_loss / len(train_loader)
                test_loss = test_loss / len(test_loader)
                accuracy_sum = accuracy / len(test_loader.dataset)

                print(f"Epoch: {epoch + 1}/{epochs} "
                      f"Train loss: {train_loss:.3f} "
                      f"Test loss: {test_loss:.3f} "
                      f"Test accuracy: {accuracy_sum * 100:.1f}%")
                running_loss = 0
                model.train()

    save_model(model, epochs, optimizer, image_datasets, save_dir)

    return save_model

# AlexNet Training Model


def model_alexnet(epochs, learn_rate, train_loader, test_loader, image_datasets, gpu, hidden_layers, save_dir):
    model = models.alexnet(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # check the model
    print("********* Model Setup Inputs***********")
    print("Epochs = {}  Learn Rate = {}  Hidden Layers = {} Model = AlexNet".format(epochs, learn_rate, hidden_layers))
    print("")
    print("**** These are the model settings for AlexNet before updating the classifier ****")
    print(model)

    # Set classifier (in features match AlexNet, hidden layers configurable)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(9216, hidden_layers)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.25)),
        ('output', nn.Linear(hidden_layers, 102)),
        ('dropout', nn.Dropout(p=0.25)),
        ('softmax', nn.LogSoftmax(dim=1))
    ]))

    # Replace classifier
    model.classifier = classifier

    # set to cuda
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()

    # Add loss calc and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    # inits
    steps = 0
    running_loss = 0
    print_every = 50

    print("********** Learn Rate= {} *********".format(learn_rate))

    # Training steps
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        for inputs, labels in train_loader:
            steps += 1

            # Move input and label tensors to the default device
            if gpu and cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero grads
            optimizer.zero_grad()

            # forward
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            # back
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        if gpu and cuda:
                            inputs = inputs.cuda()
                            labels = labels.cuda()

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.sum(equals.type(torch.FloatTensor)).item()

                train_loss = running_loss / len(train_loader)
                test_loss = test_loss / len(test_loader)
                accuracy_sum = accuracy / len(test_loader.dataset)

                print(f"Epoch: {epoch + 1}/{epochs} "
                      f"Train loss: {train_loss:.3f} "
                      f"Test loss: {test_loss:.3f} "
                      f"Test accuracy: {accuracy_sum * 100:.2f}%")
                running_loss = 0
                model.train()

    save_model(model, epochs, optimizer, image_datasets, save_dir)

    return save_model

# Saves the model, directory is configurable


def save_model(model, epochs, optimizer, image_datasets, save_dir):
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'epochs': epochs,
        'class_to_idx': model.class_to_idx,
        'optimizer': optimizer.state_dict(),
        'arch': model}

    torch.save(checkpoint, save_dir)

    return model
