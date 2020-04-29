import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random


class SegmentationDataset(Dataset):
    """
    A dataset that generates data for a pytorch model.
    """
    def __init__(self, input_folder: str, inf_type: str, img_size: int, img_transform=None, label_transform=None):
        self.input_folder = input_folder
        self.images, self.labels = self.list_files()
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.img_size = img_size
        self.inf_type = inf_type

        random.seed(self.par['random_seed'])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])

        # Data augmentation must be identical for the image and label. So we need to use the function
        # augmentation routines of Pytorch.

        width, height = image.size

        # The data augmentation needs to be performed simultanuously on input image and label
        # Here we set the parameters for cropping:
        if self.inf_type == 'train':
            patch_diff = self.img_size / 4
            crop_i = random.randint(round(-patch_diff), round(height - 3*patch_diff))
            crop_j = random.randint(round(-patch_diff), round(width - 3*patch_diff))
            crop_h = self.img_size
            crop_w = self.img_size
        else:
            crop = self.par['resize']
            crop_i = random.randint(0, round(height - self.img_size))
            crop_j = random.randint(0, round(width - self.img_size))
            crop_h = self.img_size
            crop_w = self.img_size

        # Image
        image = transforms.functional.crop(image, crop_i, crop_j, crop_h, crop_w)

        # Label
        label = transforms.functional.crop(label, crop_i, crop_j, crop_h, crop_w)

        if self.inf_type == 'train':
            if random.random() > 0.3:
                angle = random.random()*40-20
            else:
                angle = 90*random.randint(-1, 1)

            # Image
            image = transforms.functional.rotate(image, angle=angle)

            # Label
            label = transforms.functional.rotate(label, angle=angle)

        if self.img_transform:
            image = self.img_transform(image)
        if self.label_transform:
            label = self.label_transform(label)


        sample = {'image': image, 'label': label, 'filename': self.images[idx]}

        return sample

    def list_files(self) -> tuple:
        """
        Get list of all image files in the folders 'images' and 'labels'
        :return:
        """
        image_dir = os.path.join(self.input_folder, 'images')
        label_dir = os.path.join(self.input_folder, 'labels')

        for root, directories, filenames in os.walk(image_dir):
            image_list = [os.path.join(image_dir, f) for f in filenames]

        for root, directories, filenames in os.walk(label_dir):
            label_list = [os.path.join(label_dir, f) for f in filenames]

        return image_list, label_list


