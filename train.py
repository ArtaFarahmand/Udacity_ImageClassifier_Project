import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import json
from torch.autograd import variable
import argparse
import os
import sys

def args_parser():

    parser = argparse.ArgumentParser(description = 'train file')

    parser.add_argument('--data_dir', type = str, default = 'flower', help = 'dataset directory')
    parser.add_argument('--gpu', type = bool, default = 'True',  help = 'True: gpu, False: cpu')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Learning rate')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Num of epochs')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Architecture')
    parser.add_argument('--hidden_units', type = int, default = 500, help = 'Hidden unit for layer')
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Save train model to file')

    args = parser.parse_args()
    return args

def process_data(train_dir, test_dir, valid_dir):

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle = True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 64, shuffle = True)

    return trainloaders, testloaders, validloaders, train_datasets, test_datasets, valid_datasets

def basic_model(arch):

    # Load pretrained_network VGG16

    if arch == "vgg16":
        load_model = models.vgg16(pretrained = True)
        print('Using vgg16')

    elif arch == "densenet":
        load_model = models.densenet121(pretrained = True)
        print('Using densenet121')

    else:
        print('Please using vgg16 or densent, the default model is vgg16')
        load_model = models.vgg16(pretrained = True)

    return load_model

def set_classifier(load_model, hidden_units):

    if hidden_units == None:
        hidden_units = 512

    input = load_model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input, hidden_units, bias = True)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(p = 0.5)),
        ('fc2', nn.Linear(hidden_units,  128, bias = True)),
        ('relu2', nn.ReLU()),
        ('dropout', nn.Dropout(p = 0.5)),
        ('fc3', nn.Linear(128, 102, bias = True)),
        ('output', nn.LogSoftmax(dim = 1))

    ]))

    return classifier
    print(f"Epoch {epoch + 1} / {epochs}.."
          f"Train loss: {running_loss / print_every:.3f}..")

def train_model(epochs, trainloaders, validloaders, device, model, optimizer, criterion):

    if type(epochs) == type(None):
        epochs = 10
        print('Epochs = 10')

    steps = 0
    model.to(device)
    running_loss = 0
    print_every = 120
    for epoch in range(epochs):
        for inputs, labels in trainloaders:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        # Calculate accuracy

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class = labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        print(f"Epoch {epoch + 1} / {epochs}.."
                              f"Train loss: {running_loss / print_every:.3f}.."
                              f"Valid loss: {test_loss / len(validloaders):.3f}.."
                              f"Valid accuracy: {accuracy / len(validloaders):.3f}")

                    running_loss = 0
                model.train()

    return model

def valid_model(model, testloaders, device, criterion):

    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels, in testloaders:
            inputs, labels = input.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            # Calculate accuracy

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equal.type(torch.FloatTensor)).item()

    print(accuracy)

def save_checkpoint(Model, train_dataset, save_dir, arcj):
    Model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {
        'structure': arch,
        'classifier': Model.classifier,
        'state_dic': Model.state_dict(),
        'class_to_idx': Model.class_to_idx
    }

    return torch.save(checkpoint, save_dir)

def main():

    args = args_parser()
    is_gpu = args.gpu

    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    if is_gpu and use_cuda:
        device = torch.device("cuda:0")
        print(f"Device is set to {device}")

    else:
        device = torch.device("cpu")
        print(f"Device is set {device}")

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    trainloaders, testloaders, validloaders, train_datasets, test_datasets, valid_datasets = process_data(train_dir, test_dir, valid_dir)

    model = basic_model(args.arch)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = set_classifier(model, args.hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)
    trmodel = train_model(args.epochs, trainloaders, validloaders, device, model, optimizer, criterion)
    valid_model(trmodel, testloaders, device, criterion)
    save_checkpoints(trmodel, train_datasets, args.save_dir, arg.arch)

    print('Completed')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    
