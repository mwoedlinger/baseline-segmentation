import random
import os
import shutil

class DataList:
    """
    Contains information about all images and provides capabilities to perform train, eval test splits
    and automatic moving of the images to their respective folders.
    """
    def __init__(self, input_folder: str, random_seed: int=42):
        """
        The input folder where there is a folder 'images' and a folder 'labels' each
        with images with identical names
        :param input_folder: Input folder. Must contain a folder 'images' and a folder 'labels'.
        """
        random.seed(random_seed)
        self.list = self.extract_list(input_folder)

    def extract_list(self, input_folder: str) -> list:
        """
        Generates a list with all filenames
        :param input_folder: Input folder. Must contain a folder 'images' and a folder 'labels'.
        :return: A list with all filenames.
        """
        for root, directories, filenames in os.walk(input_folder):
            if (root.split(os.sep)[-1] == 'images'):
                files = [l.split('.')[0] for l in filenames]
                self.root = os.path.join(root, '..')

        return files

    def __len__(self):
        return len(self.list)

    def train_eval_test_split(self, train_frac: float) -> tuple:
        """
        Splits the list in train, eval and test set according to train_frac. train_frac is the proportion
        of training data of all data. The remaining data is split equally into validation and test data.
        :param train_frac: The proportion of training data of all data
        :return: A tuple of lists: train, validation, test
        """
        random_list = self.list.copy()
        random.shuffle(random_list)
        length = len(self.list)

        train = random_list[0:int(train_frac * length)]
        validation = random_list[(int(train_frac * length)):int((train_frac + (1 - train_frac) / 2) * length)]
        test = random_list[int((train_frac + (1 - train_frac) / 2) * length):]

        return train, validation, test

    def move_images(self, train: list, validation: list, test: list):
        """
        Moves the images from the folders 'images' and 'labels' to 'train/images', 'train/labels', 'eval/images', ...
        :param train: List with filenames
        :param validation: List with filenames
        :param test: List with filenames
        """
        folders = {'train': os.path.join(self.root, 'train'),
                   'eval': os.path.join(self.root, 'eval'),
                   'test': os.path.join(self.root, 'test')}
        files = {'train': train, 'eval': validation, 'test': test}

        file_ending_images = 'png'
        file_ending_labels = 'png'

        if not os.path.isdir(folders['train']):
            os.mkdir(os.path.join(folders['train']))
        if not os.path.isdir(folders['eval']):
            os.mkdir(os.path.join(folders['eval']))
        if not os.path.isdir(folders['test']):
            os.mkdir(os.path.join(folders['test']))

        for type in ['train', 'eval', 'test']:
            for file in files[type]:

                if not os.path.isdir(os.path.join(folders[type], 'images')):
                    os.mkdir(os.path.join(os.path.join(folders[type], 'images')))
                if not os.path.isdir(os.path.join(folders[type], 'labels')):
                    os.mkdir(os.path.join(os.path.join(folders[type], 'labels')))

                # image
                shutil.move(os.path.join(self.root, 'images', file + '.' + file_ending_images),
                            os.path.join(folders[type], 'images', file + '.' + file_ending_images))
                # label
                shutil.move(os.path.join(self.root, 'labels', file + '.' + file_ending_labels),
                            os.path.join(folders[type], 'labels', file + '.' + file_ending_labels))

        shutil.rmtree(os.path.join(self.root, 'images'))
        shutil.rmtree(os.path.join(self.root, 'labels'))