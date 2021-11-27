import pandas as pd
import torchvision
import torch
from torchvision import transforms, models
import cv2

import os
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import glob
import antialiased_cnns
import nonechucks as nc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, image_datasets, dataloaders, criterion, optimizer, num_epochs, odir):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
        os.makedirs(odir, exist_ok=True)
        torch.save(model.state_dict(), f'{odir}/weights_epoch_{epoch}.h5')
    return model


def timeit(f):
    def wrap(*args, **kwargs):
        start = pd.Timestamp.now()
        res = f(*args, **kwargs)
        elapsed = pd.Timestamp.now() - start
        print('elapsed ', elapsed)
        return res
    return wrap


@timeit
def main():
    model_name = 'resnet50'
    num_epochs = 10
    model_odir = f'trained_models/{model_name}'
    print('model_name:', model_name)

    model = antialiased_cnns.resnet50(pretrained=True).to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model._modules.get('avgpool')
    print('device:', device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
        'validation':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ]),
    }

    image_datasets = {
        'train':
            nc.SafeDataset(
                torchvision.datasets.ImageFolder('/home/tesla/tigers_bbox/princess_task_data/train',
                                                 data_transforms['train'])),
        'validation':
            nc.SafeDataset(
                torchvision.datasets.ImageFolder('/home/tesla/tigers_bbox/princess_task_data/val',
                                                 data_transforms['validation']))
    }

    dataloaders = {
        'train':
            torch.utils.data.DataLoader(image_datasets['train'],
                                        batch_size=32,
                                        shuffle=True,
                                        num_workers=0,
                                        ),
        'validation':
            torch.utils.data.DataLoader(image_datasets['validation'],
                                        batch_size=32,
                                        shuffle=False,
                                        num_workers=0
                                        )
    }

    print('model_odir', model_odir)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters())

    model_trained = train_model(model,
                                image_datasets,
                                dataloaders, criterion, optimizer, num_epochs, model_odir)


if __name__ == '__main__':
    main()
