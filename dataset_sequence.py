
import os
import random
import tensorflow as tf
import numpy as np
from keras_vggface.utils import preprocess_input

class DatasetSequence(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size, split, image_size = (224, 224), shuffle=True, augment=False):
        """
        :param dataset_path: path to the dataset, which contains the train and test folders.
        Inside each folder there are folders for each class: fake_mask, no_mask, real_mask
        :param batch_size:
        :param split: 'train', 'valid' or 'test'
        :param image_size: the size of the images to be loaded
        :param shuffle: whether to shuffle the dataset
        """
        self.train_path = os.path.join(dataset_path, 'train')
        self.test_path = os.path.join(dataset_path, 'test')
        self.val_path = os.path.join(dataset_path, 'val')
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.split = split
        self.x_train = []   # list of paths to images
        self.y_train = []   # list of labels
        self.x_test = []    # list of paths to images
        self.y_test = []    # list of labels
        self.x_val = []     # list of paths to images
        self.y_val = []     # list of labels

        # populate the lists with all the images and labels in the directory
        if self.split == 'train':
            for i, label in enumerate(os.listdir(self.train_path)): # label is the name of each folder
                for file in os.listdir(os.path.join(self.train_path, label)):
                    self.x_train.append(os.path.join(self.train_path, label, file)) # file is the name of each image
                    # The label needs to be converted to a one-hot vector
                    self.y_train.append(tf.keras.utils.to_categorical(i, num_classes=3))
        elif self.split == 'test':
            for i, label in enumerate(os.listdir(self.test_path)):
                for file in os.listdir(os.path.join(self.test_path, label)):
                    self.x_test.append(os.path.join(self.test_path, label, file))
                    self.y_test.append(tf.keras.utils.to_categorical(i, num_classes=3))
        elif self.split == 'val':
            for i, label in enumerate(os.listdir(self.val_path)):
                for file in os.listdir(os.path.join(self.val_path, label)):
                    self.x_val.append(os.path.join(self.val_path, label, file))
                    self.y_val.append(tf.keras.utils.to_categorical(i, num_classes=3))
        else:
            raise ValueError("The split must be \'train\', \'test\' or \'val\'")

        # shuffle the data
        if shuffle:
            if self.split == 'train':
                c = list(zip(self.x_train, self.y_train))
                random.shuffle(c)
                self.x_train, self.y_train = zip(*c)
            elif self.split == 'test':
                c = list(zip(self.x_test, self.y_test))
                random.shuffle(c)
                self.x_test, self.y_test = zip(*c)
            elif self.split == 'val':
                c = list(zip(self.x_val, self.y_val))
                random.shuffle(c)
                self.x_val, self.y_val = zip(*c)


    def __len__(self):
        if self.split == 'train':
            return len(self.x_train) // self.batch_size
        elif self.split == 'test':
            return len(self.x_test) // self.batch_size
        elif self.split == 'val':
            return len(self.x_val) // self.batch_size


    def __getitem__(self, idx):
        if self.split == 'train':
            batch_x = self.x_train[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_y = self.y_train[idx * self.batch_size : (idx + 1) * self.batch_size]
        elif self.split == 'test':
            batch_x = self.x_test[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_y = self.y_test[idx * self.batch_size : (idx + 1) * self.batch_size]
        elif self.split == 'val':
            batch_x = self.x_val[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_y = self.y_val[idx * self.batch_size : (idx + 1) * self.batch_size]
        else:
            raise ValueError("The split must be \'train\', \'test\' or \'val\'")

        # Return a tuple of (input,output) to feed the network
        # The first element of the tuple is the input: a numpy array of preprocessed images
        # The second element is the output: a numpy array of labels
        return self.preprocessing(np.array([self.__load_image(file_path) for file_path in batch_x])),\
               np.array(batch_y)

    def preprocessing(self, batch_images):
        # If the split is 'train', we apply data augmentation
        if self.split == 'train' and self.augment:
            # Randomly flip the images horizontally
            batch_images = tf.image.random_flip_left_right(batch_images)
            # Randomly change the brightness of the images
            batch_images = tf.image.random_brightness(batch_images, max_delta=0.5)
            # Randomly change the saturation of the images
            batch_images = tf.image.random_saturation(batch_images, lower=0, upper=2)
            # Randomly change the contrast of the images
            batch_images = tf.image.random_contrast(batch_images, lower=0.7, upper=1.3)

        # Apply the preprocessing of vgg-face
        batch_images = preprocess_input(batch_images, version=2)

        # Scale (DO NOT USE this if you use VGG-FACE,
        # since the preprocessing of vgg-face already scales the images to the required range)

        # batch_images = batch_images / 255.0

        return batch_images

    def __load_image(self, file_path):
        # Load the image with 3 channels
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=self.image_size, color_mode='rgb')
        # Convert the image to a numpy array
        img = tf.keras.preprocessing.image.img_to_array(img)
        return img
