import numpy as np
import h5py
import os
import cv2
import random
import sys

def get_pair(path, set, ids, positive):
    pair = []
    pic_name = []
    files = os.listdir('%s/%s' % (path, set))
    if positive:
        value = random.sample(ids, 1)
        id = [str(value[0]), str(value[0])]
    else:
        id = random.sample(ids, 2)
    id = [str(id[0]), str(id[1])]
    for i in range(2):
	#id_files = [f for f in files if (f[0:4] == ('%04d' % id[i]) or (f[0:2] == '-1' and id[i] == -1))]
        id_files = [f for f in files if f.split('_')[0] == id[i]]
        pic_name.append(random.sample(id_files, 1))
    for pic in pic_name:
        pair.append('%s/%s/' % (path, set) + pic[0])

    return pair

'''
def get_num_id(path, set):
    files = os.listdir('%s/%s' % (path, set))
    files.sort()
    return int(files[-1].split('_')[0]) + 1
'''

def get_id(path, set):
    files = os.listdir('%s/%s' % (path, set))
    IDs = []
    for f in files:
        IDs.append(f.split('_')[0])
    IDs = list(set(IDs))
    return IDs

def read_data(path, set, ids, image_width, image_height, batch_size):
    batch_images = []
    labels = []
    for i in range(batch_size // 2):
        pairs = [get_pair(path, set, ids, True), get_pair(path, set, ids, False)]  
        for pair in pairs:
            images = []
            for p in pair:
                image = cv2.imread(p)
                image = cv2.resize(image, (image_width, image_height))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            batch_images.append(images)
        labels.append([1., 0.])
        labels.append([0., 1.])

    '''
    for pair in batch_images:
        for p in pair:
            cv2.imshow('img', p)
            key = cv2.waitKey(0)
            if key == 1048603:
                exit()
    '''
    return np.transpose(batch_images, (1, 0, 2, 3, 4)), np.array(labels)

#if __name__ == '__main__':
#prepare_data(sys.argv[1])
