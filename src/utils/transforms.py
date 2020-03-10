import torch

class rgb_to_labels(object):
    """
    A transformer that converts a label file from a rgb image (as torch tensor) to a torch tensor containing the class
    at every point.
    """
    def __init__(self, region_types, color_dict):
        self.colors = color_dict
        self.region_types = region_types

    def __call__(self, img):
        label_list = []
        img = img.permute(1, 2, 0)*255

        for region_type in self.region_types:
            color = torch.tensor(self.colors[region_type], dtype=torch.float)
            equality = torch.eq(img, color)
            class_map = equality.all(dim=-1)
            label_list.append(class_map)

        label_img = torch.stack(label_list)

        return torch.argmax(label_img, dim=0)