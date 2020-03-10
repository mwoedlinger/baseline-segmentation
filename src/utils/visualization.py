import torch
import numpy as np

def create_color_prediction(img: torch.Tensor, colors: list):
    """
    For an input tensor with shape [1, num_classes, height, width] returns a numpy image with the most likely
    class for each pixel painted in colors specified in the list color_list
    :param img: torch tensor with shape [1, num_classes, height, width]
    :return: a numpy image
    """

    color_list = colors

    bin_array = img.detach().cpu().numpy()[0]
    bin_array = np.array([[np.eye(bin_array.shape[0])[np.argmax(bin_array[:, x, y])]
                           for x in range(0, bin_array.shape[1])]
                          for y in range(0, bin_array.shape[2])], dtype=np.uint8)

    mask = sum([np.array([bin_array[:, :, n] * color_list[n][k] for k in range(0, 3)], dtype=np.uint8) for n in
                range(0, bin_array.shape[2])])
    mask = np.transpose(mask, (0, 2, 1))

    return mask


def create_color_prediction_for_label(label: torch.Tensor, num_classes: int, colors: list):
    """
    For an input tensor with shape [height, width] where every pixel contains a number
    in {0, ..., num_classes} specifying which class is present, returns a numpy image with the most
    likely class for each pixel painted in colors specified in the list color_list
    :param img: torch tensor with shape [height, width] with values in {0, ..., num_classes}
    :return: a numpy image
    """

    color_list = colors

    bin_array = label.cpu().numpy()
    bin_array = np.array([[np.eye(num_classes)[bin_array[x, y]]
                           for x in range(0, bin_array.shape[0])]
                          for y in range(0, bin_array.shape[1])], dtype=np.uint8)

    mask = sum([np.array([bin_array[:, :, n] * color_list[n][k] for k in range(0, 3)], dtype=np.uint8) for n in
                range(0, bin_array.shape[2])])
    mask = np.transpose(mask, (0, 2, 1))

    return mask