import numpy as np
import h5py
import os
import cv2
import random

def prepare_data(path):
    f = h5py.File('%s/cuhk-03.mat' % path)
    labeled = [f['labeled'][0][i] for i in xrange(len(f['labeled'][0]))]
    labeled = [f[labeled[0]][i] for i in xrange(len(f[labeled[0]]))]
    detected = [f['detected'][0][i] for i in xrange(len(f['detected'][0]))]
    detected = [f[detected[0]][i] for i in xrange(len(f[detected[0]]))]
    datasets = [['labeled', labeled], ['detected', detected]]
    prev_id = 0

    for dataset in datasets:
        if not os.path.exists('%s/%s/train' % (path, dataset[0])):
            os.makedirs('%s/%s/train' % (path, dataset[0]))
        if not os.path.exists('%s/%s/val' % (path, dataset[0])):
            os.makedirs('%s/%s/val' % (path, dataset[0]))

        for i in xrange(0, len(dataset[1])):
            for j in xrange(len(dataset[1][0])):
                try:
                    image = np.array(f[dataset[1][i][j]]).transpose((2, 1, 0))
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image = cv2.imencode('.jpg', image)[1].tostring()
                    if len(dataset[1][0]) - j <= 100:
                        filepath = '%s/%s/val/%04d_%02d.jpg' % (path, dataset[0], j - prev_id - 1, i)
                    else:
                        filepath = '%s/%s/train/%04d_%02d.jpg' % (path, dataset[0], j, i)
                        prev_id = j
                    with open(filepath, 'wb') as image_file:
                        image_file.write(image)
                except Exception as e:
                    print('invalid data: ', i, j, str(e))
                    continue

def get_pair(path, set, num_id, positive):
    pair = []
    if positive:
        id = int(random.random() * num_id)
    for i in xrange(2):
        filepath = ''
        if not positive:
            id = int(random.random() * num_id)

        while True:
            index = int(random.random() * 10)
            filepath = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, id, index)
            if not os.path.exists(filepath):
                continue
            break
        pair.append(filepath)
    return pair

def get_num_id(path, set):
    files = os.listdir('%s/labeled/%s' % (path, set))
    files.sort()
    return int(files[-1].split('_')[0]) - int(files[0].split('_')[0]) + 1

def read_data(path, set, num_id, image_width, image_height, batch_size):
    batch_images = []
    for i in xrange(batch_size // 2):
        pairs = [get_pair(path, set, num_id, True), get_pair(path, set, num_id, False)]
        for pair in pairs:
            images = []
            for p in pair:
                image = cv2.imread(p)
                image = cv2.resize(image, (image_width, image_height))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            batch_images.append(images)

    labels = [[1., 0.] for i in xrange(batch_size)]
    '''
    for pair in batch_images:
        for p in pair:
            cv2.imshow('img', p)
            key = cv2.waitKey(0)
            if key == 1048603:
                exit()
    '''
    return np.transpose(batch_images, (1, 0, 2, 3, 4)), labels

#num_id = get_num_id('/home/thomas/Downloads/dl/dataset/cuhk03', 'val')
#read_data('/home/thomas/Downloads/dl/dataset/cuhk03', 'val', num_id, 60, 160, 32)