import os


folders = os.listdir('./dataset/')
counter = 0

folders = sorted(folders)

with open('labels.txt', 'w') as file:
    for folder in folders:
        file.write(str(counter) + ',' + folder + '\n')
        counter += 1
