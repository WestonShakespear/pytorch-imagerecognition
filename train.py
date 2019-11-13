import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, datasets
from time import time

#image properties
datasetLocation = './dataset/'

imageX = 640
imageY = 480

randomvertflip = 0.5
randomhorzflip = 0.5

randomrot = (-45, 45)

#model properties
input_size = imageX * imageY
hidden_sizes = [1024, 512, 512, 64]
output_size = 30

#training properties
learningrate = 0.003
momentum = 0.8

trainingpasses = 20


#Transform for the image
transform = [
    transforms.Compose([
        transforms.Resize((imageX, imageY)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    transforms.Compose([
        transforms.Resize((imageX, imageY)),
        transforms.Grayscale(),
        transforms.randomVerticalFlip(randomvertflip),
        transforms.randomHorizontalFlip(randomhorzflip),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    transforms.Compose([
        transforms.Resize((imageX, imageY)),
        transforms.Grayscale(),
        transforms.randomRotation(randomrot),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    transforms.Compose([
        transforms.Resize((imageX, imageY)),
        transforms.Grayscale(),
        transforms.randomVerticalFlip(randomvertflip),
        transforms.randomHorizontalFlip(randomhorzflip),
        transforms.randomRotation(randomrot),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
]

trainingset = datasets.ImageFolder(datasetLocation, transform=transform[0])

trainloader = torch.utils.data.DataLoader(trainingset, batch_size=32, shuffle=True)

print('[INFO] Loaded Data]')

model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
    nn.Linear(hidden_sizes[1], hidden_sizes[2]), nn.ReLU(),
    nn.Linear(hidden_sizes[2], hidden_sizes[3]), nn.ReLU(),
    nn.Linear(hidden_sizes[3], output_size),
    nn.LogSoftmax(dim=1)
).to("cuda")

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=learningrate, momentum=momentum)

time0 = time()

print('[INFO] Created Model')

for e in range(0, trainingpasses):
    running_loss = 0

    for images, labels in trainloader:
        images = images.to("cuda")
        labels = labels.to("cuda")

        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        #cleanup
        del images
        del labels

        torch.cuda.empty_cache()

    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        print("\nTraining Time (in minutes) =",(time()-time0)/60)

        torch.save(model, './model.pt')
