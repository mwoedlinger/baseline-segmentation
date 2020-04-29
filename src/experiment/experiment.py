import os
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time
import copy
import random
from torchvision import transforms, models
from tqdm import tqdm
import math
from ..data.dataset import SegmentationDataset
from ..models.unet_model import UNet
from ..models.gcn_model import GCN
from ..models.tiramisu_model import *
from ..models.deeplabv3_plus import DeepLab
from ..utils.transforms import rgb_to_labels
from ..utils.visualization import *
from ..utils.utils import load_class_dict
from .metrics import *
from .losses import *
import PIL


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class SegmentationExperiment:
    """
    The experiment class contains the functionalities for training and testing.
    """
    def __init__(self, cfg_exp_name, cfg_gpu, cfg_model_name, cfg_img_size, cfg_lr,
                 cfg_data_folder, cfg_batch_size, cfg_epochs, cfg_output_folder):

        self.cfg_exp_name = cfg_exp_name
        self.cfg_gpu = cfg_gpu
        self.cfg_model_name = cfg_model_name
        self.cfg_img_size = cfg_img_size
        self.cfg_lr = cfg_lr
        self.cfg_data_folder = cfg_data_folder
        self.cfg_batch_size = cfg_batch_size
        self.cfg_epochs = cfg_epochs
        self.cfg_output_folder = cfg_output_folder

        self.region_types, self.colors, self.color_dict = load_class_dict(os.path.join(cfg_data_folder, 'classes.txt'))
        self.cfg_classes = len(self.region_types)

        # For the mean IoU score:
        self.eval_labels = [n for n in range(1, len(self.region_types))]

        # self.device = torch.device('cpu')#torch.device('cuda:' + self.cfg_gpu if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda:' + self.cfg_gpu if torch.cuda.is_available() else 'cpu')
        self.model, self.criterion = self.get_model()
        self.optimizer, self.scheduler = self.get_optimizer()
        self.datasets, self.dataloaders = self.get_dataloaders()

        print('\n######################################')
        print('## exp:       ' + str(self.cfg_exp_name))
        print('## device:    ' + str(self.device))
        print('## Model:     ' + str(self.cfg_model_name))
        print('## img_size:  ' + str(self.cfg_img_size))
        print('## Crop size: ' + str(self.data_augmentation['crop']))
        print('######################################\n')

        self.dataset_sizes = {inf_type: len(self.datasets[inf_type]) for inf_type in ['train', 'eval', 'test']}

    def get_dataloaders(self):
        """
        Generates the dataloaders as a dictionary with the keys train, eval and test.
        :return: a list containing the datasets and the the dataloaders
        """
        data_transform = {'train':
                              {'images': transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize(mean=[0.7219, 0.6874, 0.6260],
                                                                                  std=[0.2174, 0.2115, 0.1989])
                                                        ]),
                              'labels': transforms.Compose([transforms.ToTensor(),
                                                            rgb_to_labels(region_types=self.region_types,
                                                                          color_dict=self.color_dict)
                                                            ])
                              },
                          'eval':
                              {'images': transforms.Compose([transforms.Resize((self.cfg_img_size, self.cfg_img_size),
                                                                               interpolation=PIL.Image.NEAREST),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(mean=[0.7219, 0.6874, 0.6260],
                                                                                  std=[0.2174, 0.2115, 0.1989])
                                                        ]),
                              'labels': transforms.Compose([transforms.Resize((self.cfg_img_size, self.cfg_img_size),
                                                                              interpolation=PIL.Image.NEAREST),
                                                            transforms.ToTensor(),
                                                            rgb_to_labels(region_types=self.region_types,
                                                                          color_dict=self.color_dict)
                                                            ])
                              },
                          'test':
                              {'images': transforms.Compose([transforms.Resize((self.cfg_img_size, self.cfg_img_size),
                                                                               interpolation=PIL.Image.NEAREST),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(mean=[0.7219, 0.6874, 0.6260],
                                                                                  std=[0.2174, 0.2115, 0.1989])
                                                        ]),
                              'labels': transforms.Compose([transforms.Resize((self.cfg_img_size, self.cfg_img_size),
                                                                              interpolation=PIL.Image.NEAREST),
                                                            transforms.ToTensor(),
                                                            rgb_to_labels(region_types=self.region_types,
                                                                          color_dict=self.color_dict)
                                                            ])
                              }}
        shuffle = {'train': True, 'eval': False, 'test': False}
        batch_size_dict = {'train': self.cfg_batch_size, 'eval': 4, 'test': 4}

        image_datasets = {inf_type: SegmentationDataset(input_folder=os.path.join(self.cfg_data_folder, inf_type),
                                                        inf_type=inf_type,
                                                        data_augmentation_par=self.cfg_img_size,
                                                        img_transform=data_transform[inf_type]['images'],
                                                        label_transform=data_transform[inf_type]['labels'])
                          for inf_type in ['train', 'eval', 'test']}

        dataloaders = {inf_type: torch.utils.data.DataLoader(image_datasets[inf_type],
                                                             batch_size=batch_size_dict[inf_type],
                                                             shuffle=shuffle[inf_type],
                                                             num_workers=4)
                       for inf_type in ['train', 'eval', 'test']}

        return [image_datasets, dataloaders]


    def get_model(self):
        """
        Load the correct model with pretrained imageNet weights.
        :return: a list containing the model and the loss function (CrossEntropyLoss)
        """

        if self.cfg_model_name == 'DeepLabV3_resnet50':
            model_ft = models.segmentation.deeplabv3_resnet50(pretrained=False,
                                                              progress=True,
                                                              num_classes=self.cfg_classes)
        elif self.cfg_model_name == 'DeepLabV3_resnet101':
            model_ft = models.segmentation.deeplabv3_resnet101(pretrained=False,
                                                               progress=True,
                                                               num_classes=self.cfg_classes)
        elif self.cfg_model_name == 'FCN_resnet50':
            model_ft = models.segmentation.fcn_resnet50(pretrained=False,
                                                        progress=True,
                                                        num_classes=self.cfg_classes)
        elif self.cfg_model_name == 'FCN_resnet101':
            model_ft = models.segmentation.fcn_resnet101(pretrained=True,
                                                        progress=True,
                                                        num_classes=self.cfg_classes)
        elif self.cfg_model_name == 'GCN':
            model_ft = GCN(n_classes=self.cfg_classes)
            raise NotImplementedError

        model_ft = model_ft.to(self.device)

        criterion = nn.CrossEntropyLoss()

        # loss_weighting = torch.tensor([0.9, 1.0, 1.0, 1.0, 1.2, 1.2]).to(self.device)
        # criterion = nn.CrossEntropyLoss(weight=loss_weighting)

        # criterion = FocalLoss(gamma=0.5)

        # criterion = CEDiceLoss()

        return [model_ft, criterion]


    def get_optimizer(self):
        """
        Get the optimizer and scheduler.
        :return: a list containing the optimizer and the scheduler
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg_lr)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg_lr)
        # optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.cfg_lr, weight_decay=2e-5, momentum=0.9)

        # Decay LR by a factor of 'gamma' every 'step_size' epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

        return [optimizer, exp_lr_scheduler]


    def set_mode(self, mode: str):
        """
        Sets the mode to 'train' or 'eval'
        """
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise NotImplementedError


    def load_model(self, model_path: str):
        """
        Loads the saved model 'model_path'
        """
        self.model.load_state_dict(torch.load(model_path), map_location=self.device).to(self.device)

    def save_model(self):
        """
        Saves the model in the directory cfg_output_folder with the name self.cfg_exp_name.pt
        """
        torch.save(self.model.state_dict(), os.path.join(self.cfg_output_folder, self.cfg_exp_name + '.pt'))


    def train_model(self):
        """
        Performs the training loop. During the training it writes the tensorboard log in /runs.
        It saves the model everytime it improves in cfg_output_folder.
        :return: the trained model
        """
        writer = SummaryWriter(log_dir=os.path.join('logs', self.cfg_exp_name))

        num_epochs = self.cfg_epochs

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        global_step_counter = 0

        # The training loop
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'eval']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                batch_counter = 0
                running_miou = []
                iou_dict = {k: [] for k in self.eval_labels}
                accuracy_dict = {k: [] for k in self.eval_labels}
                precision_dict = {k: [] for k in self.eval_labels}
                recall_dict = {k: [] for k in self.eval_labels}
                f1score_dict = {k: [] for k in self.eval_labels}

                # Iterate over data.
                for batch in tqdm(self.dataloaders[phase]):
                    # Otherwise batchnorm complains:
                    if phase == 'train' and len(batch['image']) == 1:
                        continue

                    batch_counter += 1
                    global_step_counter += self.cfg_batch_size

                    inputs = batch['image']
                    labels = batch['label']
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history only if in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if self.cfg_model_name == 'GCN':
                            outputs = self.model(inputs)[0]
                        elif self.cfg_model_name in ['UNet', 'DeepLabV3_plus', 'Tiramisu']:
                            outputs = self.model(inputs)
                        else:
                            outputs = self.model(inputs)['out']

                        preds = torch.argmax(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    running_corrects += \
                        torch.sum(preds == labels).double()/(labels.shape[0]*self.cfg_img_size*self.cfg_img_size)

                    # Computer scores
                    if phase == 'eval':
                        for n in range(0, len(labels)):
                            new_iou = miou(outputs[n], labels[n], self.eval_labels)
                            running_miou.append(np.mean(list(new_iou.values())))
                            for k in new_iou.keys():
                                iou_dict[k].append(new_iou[k])

                            a, p, r, f = compute_scores(outputs[n], labels[n], self.eval_labels)
                            for k in a.keys():
                                if not math.isnan(a[k]):
                                    accuracy_dict[k].append(a[k])
                            for k in p.keys():
                                if not math.isnan(p[k]):
                                    precision_dict[k].append(p[k])
                            for k in r.keys():
                                if not math.isnan(r[k]):
                                    recall_dict[k].append(r[k])
                            for k in f.keys():
                                if not math.isnan(f[k]):
                                    f1score_dict[k].append(f[k])


                    # Write tensorboard logs. For train mode write after every step, for eval after every epoch
                    if phase == 'train':
                        # Write Tensorboard log files
                        writer.add_scalar(tag='loss/train', scalar_value=running_loss/batch_counter,
                                          global_step=global_step_counter)
                        writer.add_scalar(tag='acc/train', scalar_value=running_corrects/batch_counter,
                                          global_step=global_step_counter)

                # Compute epoch loss and acc averages
                epoch_loss = running_loss / batch_counter
                epoch_acc = running_corrects / batch_counter

                # Write tensorboard logs and print mIoU scores
                if phase in ['eval']:
                    print(self.cfg_exp_name)
                    print('class wise scores:')
                    IoU_scores = []
                    for l in self.eval_labels:
                        writer.add_scalar(tag='miou_'+self.region_types[l]+'/'+str(phase),
                                          scalar_value=(np.sum(iou_dict[l])/len(iou_dict[l])),
                                          global_step=global_step_counter)
                        IoU_scores.append(np.sum(iou_dict[l])/len(iou_dict[l]))
                        print('{:25s} {}:{:4.3f}, {}:{:4.3f}, {}:{:4.3f}, {}:{:4.3f}, {}:{:4.3f}, '.format(self.region_types[l],
                                                                               'IoU', (np.sum(iou_dict[l])/len(iou_dict[l])),
                                                                               'acc', (np.sum(accuracy_dict[l])/len(accuracy_dict[l])),
                                                                               'pre', (np.sum(precision_dict[l])/len(precision_dict[l])),
                                                                               'rec', (np.sum(recall_dict[l])/len(recall_dict[l])),
                                                                               'f1s', (np.sum(f1score_dict[l])/len(f1score_dict[l]))))
                    print(' ')
                    print('{:15s} {:4.3f}'.format('total miou:', np.mean(running_miou)))
                    print('{:15s} {:4.3f}'.format('mIoU:', np.mean(IoU_scores)))
                    print('{:15s} {:4.3f}'.format('lr: ', get_lr(self.optimizer)))

                    writer.add_scalar(tag='mIoU/mIoU_'+phase, scalar_value=np.mean(IoU_scores),
                                      global_step=global_step_counter)

                    # deep copy the model
                    if epoch_acc >= best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        self.save_model()
                        print('\n## Save model as new best')

                if phase in ['train', 'eval']:
                    # Write Tensorboard log files
                    writer.add_scalar(tag='loss/'+phase, scalar_value=epoch_loss,
                                      global_step=global_step_counter)
                    writer.add_scalar(tag='acc/'+phase, scalar_value=epoch_acc,
                                      global_step=global_step_counter)
                    writer.add_scalar(tag='mIoU/RmIoU_'+phase, scalar_value=np.mean(running_miou),
                                      global_step=global_step_counter)
                    writer.add_scalar(tag='lr', scalar_value=get_lr(self.optimizer),
                                      global_step=global_step_counter)

                    writer.add_image(tag=phase+'/label',
                                     img_tensor=create_color_prediction_for_label(batch['label'][0],
                                                                                  num_classes=self.cfg_classes,
                                                                                  colors=self.colors),
                                     global_step=global_step_counter)
                    writer.add_image(tag=phase+'/pred', img_tensor=create_color_prediction(outputs, self.colors),
                                     global_step=global_step_counter)

                # Print scores
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # Apply learning rate scheduler
                if phase == 'train':
                    self.scheduler.step()

        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best eval Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        print('best acc:    ' + str(best_acc))
        print('current acc: ' + str(epoch_acc))

        writer.close()

        return self.model


    def test_model(self):
        """
        Test the model on the test set. Prints the output of the evaluation to the console.
        """
        self.model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        running_miou = []
        iou_dict = {k: [] for k in self.eval_labels}
        accuracy_dict = {k: [] for k in self.eval_labels}
        precision_dict = {k: [] for k in self.eval_labels}
        recall_dict = {k: [] for k in self.eval_labels}
        f1score_dict = {k: [] for k in self.eval_labels}

        # Iterate over data.
        for batch in tqdm(self.dataloaders['test']):
            inputs = batch['image']
            labels = batch['label']

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                if self.cfg_model_name == 'GCN':
                    outputs = self.model(inputs)[0]
                elif self.cfg_model_name in ['UNet', 'DeepLabV3_plus', 'Tiramisu']:
                    outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)['out']
                preds = torch.argmax(outputs, 1)
                loss = self.criterion(outputs, labels)

            for n in range(0, len(labels)):
                new_iou = miou(outputs[n], labels[n], self.eval_labels)
                running_miou.append(np.mean(list(new_iou.values())))
                for k in new_iou.keys():
                    iou_dict[k].append(new_iou[k])

                a, p, r, f = compute_scores(outputs[n], labels[n], self.eval_labels)
                for k in a.keys():
                    if not math.isnan(a[k]):
                        accuracy_dict[k].append(a[k])
                for k in p.keys():
                    if not math.isnan(p[k]):
                        precision_dict[k].append(p[k])
                for k in r.keys():
                    if not math.isnan(r[k]):
                        recall_dict[k].append(r[k])
                for k in f.keys():
                    if not math.isnan(f[k]):
                        f1score_dict[k].append(f[k])

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += \
                torch.sum(preds == labels).double()/(labels.shape[0]*self.cfg_img_size*self.cfg_img_size)

        epoch_loss = running_loss / self.dataset_sizes['test']
        epoch_acc = running_corrects.double() / self.dataset_sizes['test']

        print(self.cfg_exp_name)
        print('class wise scores:')
        IoU_scores = []
        for l in self.eval_labels:
            IoU_scores.append(np.sum(iou_dict[l]) / len(iou_dict[l]))
            print('{:25s} {}:{:4.3f}, {}:{:4.3f}, {}:{:4.3f}, {}:{:4.3f}, {}:{:4.3f}, '.format(self.region_types[l],
                                                                                               'IoU', (np.sum(iou_dict[l]) / len(iou_dict[l])),
                                                                                               'acc', (np.sum(accuracy_dict[l]) / len(accuracy_dict[l])),
                                                                                               'pre', (np.sum(precision_dict[l]) / len(precision_dict[l])),
                                                                                               'rec', (np.sum(recall_dict[l]) / len(recall_dict[l])),
                                                                                               'f1s', (np.sum(f1score_dict[l]) / len(f1score_dict[l]))))
        print(' ')
        print('{:15s} {:4.3f}'.format('total miou:', np.mean(running_miou)))
        print('{:15s} {:4.3f}'.format('mIoU:', np.mean(IoU_scores)))

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', epoch_loss, epoch_acc))

