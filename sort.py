import os
from shutil import copyfile

input = './data-pre/'
output = './dataset/'

folders = os.listdir(input)

dirs = []

for folder in folders:
    csv = input + folder + '/csv/metadata.csv'
    imagefolder = input + folder + '/images/'

    with open(csv, 'r') as file:
        for line in file:
            line = line.split(',')

            imagename = line[0]
            type = line[3]
            brand = line[4]

            iden = type.replace(' ', '')

            if iden not in dirs:
                os.makedirs(output + iden)
                dirs.append(iden)

            copyfile(input + folder + '/images/' + imagename, output + iden + '/' + imagename)
