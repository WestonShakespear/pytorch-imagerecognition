import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import os

def evaluate(image, labels, model):
    with torch.no_grad():
        logps = model(image.view(1, 205140))

    ps = torch.exp(logps)
    ps_cpu = ps.to("cpu")

    probab = list(ps_cpu.numpy()[0])

    prediction = probab.index(max(probab))

    return labels[prediction]



transform = transforms.Compose([transforms.Resize((526, 390)), transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),])

imagename = './image.jpg'

image = Image.open(imagename)

image_tensor = transform(image)
image_tensor = image_tensor.to("cuda")

model = torch.load('./model.pt')
model = model.to("cuda")

labels = []

with open('labels.txt', 'r') as file:
    for line in file:
        labels.append(line[:-1].split(',')[1])

input = './data-pre/'

folders = os.listdir(input)

dirs = []

correct = 0
total = 0
average_time = 0

for folder in folders:
    csv = input + folder + '/csv/metadata.csv'
    imagefolder = input + folder + '/images/'

    with open(csv, 'r') as file:
        for line in file:
            line = line.split(',')

            imagename = line[0]
            type = line[3]

            image = transform(Image.open(input + folder + '/images/' + imagename)).to("cuda")


            time0 = time()
            predic = evaluate(image, labels, model)
            time_lapsed = (time() - time0)

            if predic == type.replace(' ', ''):
                correct += 1

            total += 1

            average_time = average_time + time_lapsed

            if total % 100 == 0:
                print(total)
            torch.cuda.empty_cache()

            if total > 7000:
                break

print(average_time / total)

print(str(correct / total))
