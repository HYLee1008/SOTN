import numpy as np
import os
from src.youtubeface import load_ytf_pairs
from src.lfw import load_lfw_pairs
from src.facescrub import load_fs_pairs


class wrapper_basicImg():
    def __init__(self, dataset):
        self.img_shape = [112, 112, 3]

        self.images={}
        self.labels={}

        if dataset == "YTF":
            print("Evaluation Dataset: Youtube Face Dataset")

            basedir = '/home/hy/Dataset/face/Youtube Face/detected_faces'
            pair_txt = '/home/hy/Dataset/face/Youtube Face/splits.txt'

            if os.path.exists('./npy/YTF_embedding.npy'):
                self.images, self.labels = np.load('./npy/YTF_embedding.npy'), np.load('./npy/YTF_labels.npy')

                print("Evaluation dataset loaded")
            else:
                self.images, self.labels = load_ytf_pairs(basedir, pair_txt, 1)
                np.save('./npy/YTF_embedding.npy', self.images)
                np.save('./npy/YTF_labels.npy', self.labels)

                print("Evaluation dataset newly created")

        elif dataset == "LFW":
            print("Evaluation Dataset: LFW Dataset")

            if os.path.exists('./npy/LFW_embedding.npy'):
                self.images, self.labels = np.load('./npy/LFW_embedding.npy'), np.load('./npy/LFW_labels.npy')

                print("Evaluation dataset loaded")
            else:
                self.images, self.labels = load_lfw_pairs()

                np.save('./npy/LFW_embedding.npy', self.images)
                np.save('./npy/LFW_labels.npy', self.labels)

                print("Evaluation dataset newly created")

        elif dataset == "FS":
            print("Evaluation Dataset: FS Dataset")

            if os.path.exists('./npy/FS_embedding.npy'):
                self.images, self.labels = np.load('./npy/FS_embedding.npy'), np.load('./npy/FS_labels.npy')

                print("Evaluation dataset loaded")
            else:
                self.images, self.labels = load_fs_pairs(5000)

                np.save('./npy/FS_embedding.npy', self.images)
                np.save('./npy/FS_labels.npy', self.labels)

                print("Evaluation dataset newly created")


        self.num_samples = len(self.images)

        self.next_batch_pointer = 0

        # for test inference
        self.samples_left = self.num_samples

        self.next_batch_pointer = 0


    def get_next_batch(self, batch_size):
        num_samples_left = self.num_samples - self.next_batch_pointer


        if num_samples_left >= batch_size:
            batch = self.images[self.next_batch_pointer:self.next_batch_pointer + batch_size]
            batch_label = self.labels[self.next_batch_pointer:self.next_batch_pointer + batch_size]


            self.next_batch_pointer += batch_size
        else:
            batch = self.images[self.next_batch_pointer:]
            batch_label= self.labels[self.next_batch_pointer:]


        self.samples_left = num_samples_left - batch_size


        return batch, batch_label