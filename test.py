import os
from sacred import Experiment
from src.experiment.experiment import SegmentationExperiment

ex = Experiment('segmentation_TU_library_cards')


@ex.config
def cgf():
    """
    Config function for the sacred experiment.
    """
    cfg_exp_name = 'GCN_1609_1024'
    cfg_gpu = '3'
    cfg_model_name = 'GCN'
    cfg_img_size = 1024
    cfg_classes = 9
    cfg_lr = 4e-4
    cfg_data_folder = os.path.join('data', 'segmentation', 'sets', 'combined')
    cfg_output_folder = os.path.join('trained_models', 'segmentation', cfg_model_name)
    cfg_batch_size = 2
    cfg_epochs = 60
    cfg_label_values = [[0, 0, 255], [255, 155, 0], [0, 255, 255], [128, 0, 128],
                        [0, 255, 0], [255, 255, 0], [50, 50, 50], [0, 0, 0], [255, 255, 255]]


@ex.automain
def test(cfg_exp_name, cfg_gpu, cfg_model_name, cfg_img_size, cfg_classes, cfg_lr,
          cfg_data_folder, cfg_batch_size, cfg_epochs, cfg_output_folder, cfg_label_values):
    """
    Trains the model.
    :param cfg_exp_name:    Name of the experiment
    :param cfg_gpu:         gpu that should be used
    :param cfg_model_name:  Name of the model
    :param cfg_img_size:    Image size. (cfg_img_size x cfg_img_size)
    :param cfg_classes:     Number of classes
    :param cfg_lr:          Learning rate
    :param cfg_data_folder: Folder that contains train, test and eval data
    :param cfg_batch_size:  Batch size
    :param cfg_epochs:      Number of epochs
    """

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg_gpu

    if not os.path.isdir(os.path.join(cfg_output_folder)):
        os.mkdir(os.path.join(cfg_output_folder))

    exp = SegmentationExperiment(cfg_exp_name, cfg_gpu, cfg_model_name, cfg_img_size, cfg_classes,
                                 cfg_lr, cfg_data_folder, cfg_batch_size, cfg_epochs,
                                 cfg_output_folder, cfg_label_values)

    print('## Load model')
    exp.load_model(os.path.join('trained_models', 'segmentation', cfg_model_name, cfg_exp_name))

    print('## Test model')
    exp.test_model()