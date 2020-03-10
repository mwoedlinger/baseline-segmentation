import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random


class SegmentationDataset(Dataset):
    """
    A dataset that generates data for a pytorch model.
    """
    def __init__(self, input_folder: str, inf_type: str, data_augmentation_par: dict, img_transform=None, label_transform=None):
        self.input_folder = input_folder
        self.images, self.labels = self.list_files()
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.par = data_augmentation_par
        self.inf_type = inf_type

        random.seed(self.par['random_seed'])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])

        # Data augmentation must be identical for the image and label. So we need to use the function
        # augmentation routines of Pytorch.
        # Only apply data augmentation for image, label pairs from the train set:
        if self.inf_type == 'train':

            image = transforms.Resize((self.par['resize'], self.par['resize']), interpolation=Image.NEAREST)(image)
            label = transforms.Resize((self.par['resize'], self.par['resize']), interpolation=Image.NEAREST)(label)

            augment_rand = random.randint(0, 5)


            if self.par['apply_crop'] and augment_rand > 3:
                # The data augmentation needs to be performed simultanuously on input image and label
                # Here we set the parameters for cropping:
                crop = random.randint(self.par['crop'], self.par['resize'])
                # crop_i = random.randint(0, max(self.par['resize']-crop-1, 0))
                # crop_j = random.randint(0, max(self.par['resize']-crop-1, 0))
                patch_diff = crop / 2
                crop_i = random.randint(round(-patch_diff/2), round(self.par['resize'] - crop + patch_diff/2))
                crop_j = random.randint(round(-patch_diff/2), round(self.par['resize'] - crop + patch_diff/2))
                crop_h = crop# + crop_i
                crop_w = crop# + crop_j

                # Image
                image = transforms.functional.crop(image, crop_i, crop_j, crop_h, crop_w)
                image = transforms.Resize((self.par['resize'], self.par['resize']),
                                          interpolation=Image.NEAREST)(image)

                # Label
                label = transforms.functional.crop(label, crop_i, crop_j, crop_h, crop_w)
                label = transforms.Resize((self.par['resize'], self.par['resize']),
                                          interpolation=Image.NEAREST)(label)

            if self.par['apply_affine'] and augment_rand in [2,3,4]:
                # The data augmentation needs to be performed simultanuously on input image and label
                # Here we set the parameters for the affine transformation:
                # angle = random.uniform(-self.par['affine']['angle'], self.par['affine']['angle'])
                angle = random.uniform(-self.par['affine']['angle'], self.par['affine']['angle'])
                translateH = random.uniform(self.par['affine']['translate'][0], self.par['affine']['translate'][1])
                translateV = random.uniform(self.par['affine']['translate'][0], self.par['affine']['translate'][1])
                scale = random.uniform(self.par['affine']['scale'][0], self.par['affine']['scale'][1])
                shear = random.uniform(0, self.par['affine']['shear'])
                fillcolor = self.par['affine']['fillcolor']

                # Image
                image = transforms.functional.affine(image, angle=angle, translate=(translateH, translateV),
                                                     scale=scale, shear=shear, fillcolor=fillcolor)
                # image = transforms.functional.rotate(image, angle=angle)

                # Label
                label = transforms.functional.affine(label, angle=angle, translate=(translateH, translateV),
                                                     scale=scale, shear=shear, fillcolor=fillcolor)
                # label = transforms.functional.rotate(label, angle=angle)

            # TODO: Check how model performs if only rotations with 0, 90, 180 and 270 degrees are performed
            if self.par['apply_affine'] and augment_rand < 2:
                angle = 90*random.randint(-1, 1)
                # Image
                image = transforms.functional.rotate(image, angle=angle)
                # Label
                label = transforms.functional.rotate(label, angle=angle)

        if self.img_transform:
            image = self.img_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        sample = {'image': image, 'label': label}

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
            label_list = [os.path.join(label_dir, f) for f in filenames]

        return image_list, label_list


