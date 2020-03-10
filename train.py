import os
from sacred import Experiment
from src.experiment.experiment import SegmentationExperiment

ex = Experiment('segmentation_TU_library_cards')


@ex.config
def cgf():
    """
    Config function for the sacred experiment.
    """
    cfg_model_name = 'GCN'
    cfg_img_size = 1024
    cfg_exp_name = 'v13_'+cfg_model_name + '_' + str(cfg_img_size) + '_btsb_nn_upscaling'
    cfg_gpu = '3'
    cfg_lr = 6e-4#
    cfg_data_folder = os.path.join('data', 'cBAD_'+str(cfg_img_size)+'_btsb')
    cfg_output_folder = os.path.join('trained_models', cfg_model_name)
    cfg_batch_size = 3
    cfg_epochs = 120


@ex.automain
def train(cfg_exp_name, cfg_gpu, cfg_model_name, cfg_img_size, cfg_lr,
          cfg_data_folder, cfg_batch_size, cfg_epochs, cfg_output_folder):
    """
    Trains the model.
    :param cfg_exp_name:    Name of the experiment
    :param cfg_gpu:         gpu that should be used
    :param cfg_model_name:  Name of the model
    :param cfg_img_size:    Image size. (cfg_img_size x cfg_img_size)
    :param cfg_lr:          Learning rate
    :param cfg_data_folder: Folder that contains train, test and eval data
    :param cfg_batch_size:  Batch size
    :param cfg_epochs:      Number of epochs
    """

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg_gpu

    if not os.path.isdir(os.path.join(cfg_output_folder)):
        os.mkdir(os.path.join(cfg_output_folder))

    exp = SegmentationExperiment(cfg_exp_name, cfg_gpu, cfg_model_name, cfg_img_size, cfg_lr,
                                 cfg_data_folder, cfg_batch_size, cfg_epochs, cfg_output_folder)

    print('## Train model')
    exp.train_model()

    print('## Save model')
    exp.save_model()

    print('## Test model')
    exp.test_model()