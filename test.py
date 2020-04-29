import argparse
import json
from src.experiment.experiment import SegmentationExperiment

def test(config, weights=None):
    """
    Trains the model.
    cfg_exp_name:    Name of the experiment
    cfg_gpu:         gpu that should be used
    cfg_model_name:  Name of the model
    cfg_img_size:    Image size. (cfg_img_size x cfg_img_size)
    cfg_lr:          Learning rate
    cfg_data_folder: Folder that contains train, test and eval data
    cfg_batch_size:  Batch size
    cfg_epochs:      Number of epochs
    """
    exp = SegmentationExperiment(config['cfg_exp_name'],
                                 config['cfg_gpu'],
                                 config['cfg_model_name'],
                                 config['cfg_img_size'],
                                 config['cfg_lr'],
                                 config['cfg_data_folder'],
                                 config['cfg_batch_size'],
                                 config['cfg_epochs'],
                                 config['cfg_output_folder'])
    exp.test_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model.')
    parser.add_argument('--config', help='The config file.', required=False)
    args = vars(parser.parse_args())

    config_file = args['config']
    if config_file is None:
        config_file = 'config_' + args['model_type'] + '.json'

    print('## Load config from file: ' + str(config_file))

    with open(config_file, 'r') as json_file:
        config = json.loads(json_file.read())

    test(config=config, weights=None)