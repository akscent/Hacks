import matplotlib as plt

plt.pyplot.style.use("ggplot")
plt.use('Agg')

import torch
import torch.nn as nn
import torchvision.transforms as transforms


import timm

import gc
import os
import time
import random
from datetime import datetime

from PIL import Image
from tqdm.notebook import tqdm
from sklearn import model_selection, metrics
from pathlib import Path

dataset_path = Path('')
os.listdir(dataset_path)
train_df = pd.read_csv(dataset_path/'train.csv', sep='\t')
train_df.head()

# time_label_map = {}
# for i, row in result_df.iterrows():
#     time_label_map[row['Modification Time']] = row['label_id']

# import datetime
# root_dir = "test/"
# files = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
# 
# metadata_list = []
# for file_path in files:
#     mod_time = os.path.getmtime(file_path)
#     time = datetime.datetime.fromtimestamp(mod_time).strftime('%H:%M')
#     metadata_list.append({'Filename': file_path, 'Modification Time': time})
# 
# test_metadata = pd.DataFrame(metadata_list)
# 
# test_metadata.head()
# test_metadata['Filename'] = train_metadata['Filename'].str.replace("train/", "")
# test_df = test_metadata.copy()
# test_df['label_id'] = test_df['Modification Time'].map(time_label_map)
# test_df.head()


train_df['path'] = train_df['image_name'].map(lambda x:dataset_path/'train'/x)
train_df = train_df.drop(columns=['image_name'])
train_df = train_df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
train_df.head(10)
len_df = len(train_df)


import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import os

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import time
from tqdm import tqdm


def set_requires_grad(model, value=False):
    for param in model.parameters():
        param.requires_grad = value


def train_model(model, dataloaders, criterion, optimizer, phases, num_epochs=3):
    start_time = time.time()

    acc_history = {k: list() for k in phases}
    loss_history = {k: list() for k in phases}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n_batches = len(dataloaders[phase])
            for inputs, labels in tqdm(dataloaders[phase], total=n_batches):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()
            epoch_acc /= len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)

        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))

    return model, acc_history


def init_model(device, num_classes):
    model = timm.create_model("MaxViT-xl-tf-512", pretrained=True, in_chans=3, num_classes=num_classes)
    model = model.to(device)
    return model

class ArtDataset(Dataset):
    def __init__(self, root_dir, csv_path=None, transform=None):

        self.transform = transform
        self.files = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
        self.targets = None
        if csv_path:
            df = pd.read_csv(csv_path, sep="\t")
            self.targets = df["label_id"].tolist()
            self.files = [os.path.join(root_dir, fname) for fname in df["image_name"].tolist()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        target = self.targets[idx] if self.targets else -1
        if self.transform:
            image = self.transform(image)
        return image, target

img_dir = 'train/'

# hardcode
MODEL_WEIGHTS = "baseline.pt"
TRAIN_DATASET = "train/"
TRAIN_CSV = "train.csv"

if __name__ == "__main__":
    img_size = 224
    trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1) if len(x.shape) == 2 or x.shape[-1] == 1 else x), # if Gray => make 3 channels
        # transforms.Lambda(lambda x: x[:4, :, :]), # if 4 channels => make 3
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dset = ArtDataset(TRAIN_DATASET, TRAIN_CSV, trans)
    labels = dset.targets
    indices = list(range(len(labels)))
    # Create a KFold object and fit the data
    kf = KFold(n_splits=15)
    kf.get_n_splits(indices)
    best_val_acc = 0.0
    best_model = None

    # Loop over the folds
    for train_index, val_index in kf.split(X):
      trainset = torch.utils.data.Subset(dset, train_index)
      testset = torch.utils.data.Subset(dset, val_index)
      batch_size = 32
      num_workers = 4
      trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
      testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
      loaders = {'train': trainloader, 'val': testloader}
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      model = init_model(device, num_classes=40)
      pretrain_optimizer = torch.optim.SGD(model.classifier[3].parameters(),
                                         lr=0.001, momentum=0.9)
      train_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
      criterion = nn.CrossEntropyLoss()

    # Pretrain
    # запустить предобучение модели на две эпохи
      pretrain_results = train_model(model, loaders, criterion, pretrain_optimizer,
                                   phases=['train', 'val'], num_epochs=3)

    # Train
    # запустить дообучение модели
      set_requires_grad(model, True)
      train_results = train_model(model, loaders, criterion, train_optimizer,
                                phases=['train', 'val'], num_epochs=3)

      torch.save(model.state_dict(), MODEL_WEIGHTS)
      print("Validation Accuracy: {:.4f}".format(val_acc))
    

#################################################
from fastai.data.block import DataBlock, CategoryBlock
from fastai.tabular.all import TabularDataLoaders
import fastai
from PIL import Image
from fastai.vision.all import *

from fastai.data.block import FloatListBlock, CategoryBlock
img_size = 245
transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1) if len(x.shape) == 2 or x.shape[-1] == 1 else x), # if Gray => make 3 channels
        # transforms.Lambda(lambda x: x[:4, :, :]), # if 4 channels => make 3
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data = ImageDataLoaders.from_csv(TRAIN_DATASET, csv_labels=TRAIN_CSV, item_tfms=Resize(img_size), batch_tfms=aug_transforms(), size=224, bs=32)


data = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    item_tfms=transforms,
    batch_tfms=[*aug_transforms, *transforms],
    splitter=FuncSplitter(lambda o: random.random() < 0.2)
)

dls = data.dataloaders(path, bs=bs)

from fastai.vision.data import ImageDataLoaders

def get_transforms(img_size):
    return get_transforms(do_flip=False, flip_vert=False, max_rotate=0.0, 
                         max_zoom=1.0, max_lighting=0.0, max_warp=0.0, 
                         p_affine=0.75, p_lighting=0.75, 
                         xtra_tfms=[])

data = ImageDataLoaders.from_csv(TRAIN_DATASET, csv_labels=TRAIN_CSV, 
                                  item_tfms=get_transforms(img_size=224),
                                  batch_size=32)

learn = cnn_learner(data, models.MaxViT_xl_tf_512, metrics=fbeta)

kfolds = KFold(n_splits=15)
val_accs = []

for train_index, val_index in kfolds.split(data.x):
    fold_data = ImageDataBunch.create(data.x[train_index], data.y[train_index], data.x[val_index], data.y[val_index])
    learn.data = fold_data
    learn.fit(1)
    val_acc = learn.validate()[1].item()
    val_accs.append(val_acc)

print(f"Average validation accuracy: {np.mean(val_accs)}")
