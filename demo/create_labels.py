import os, sys
sys.path.append('..')
#sys.path.append('C:\\Users\\matthias\\Documents\\myProjects\\baseline_segmentation')

import shutil
from tqdm import tqdm
from src.utils.generate_segmentation_labels import XMLParserBaselines
from src.utils.utils import load_class_dict
from distutils.dir_util import copy_tree

max_side_length = 1024
thickness = round(2.0*max_side_length/256)-6 #-6 for 1024 #-2 for 512
dot_thickness = round(2.0*max_side_length/256)-5 #-5 for 1024 #-2 for 512
class_file = os.path.join('..', 'data', 'class_files', 'classes_btsb.txt')
classes, colors, color_dict = load_class_dict(class_file)

print('thickness: {}'.format(thickness))
print('dot_thickness: {}'.format(dot_thickness))
print('classes:\n')
for c in classes:
    print('{:20s} {}'.format(c, color_dict[c]))

cfg_input_folder = os.path.join('..', 'data', 'cBAD-ICDAR2019')
cfg_output_folder = os.path.join('..', 'data', 'cBAD_' + str(max_side_length) + '_min_2')
cfg_pad = False

if not os.path.isdir(os.path.join(cfg_output_folder)):
    os.mkdir(cfg_output_folder)
shutil.copy(class_file, os.path.join(cfg_output_folder, 'classes_btsb.txt'))

for root, directories, filenames in os.walk(cfg_input_folder):
    if (root.split(os.sep)[-1] == 'page'):
        print('Processing ' + root[:-4])
        for file in tqdm(filenames):
            if not os.path.isdir(os.path.join(cfg_output_folder, root.split(os.sep)[-2])):
                os.mkdir(os.path.join(cfg_output_folder, root.split(os.sep)[-2]))
            xml_parser = XMLParserBaselines(xml_filename=os.path.join(root, file),
                                            input_folder=root[:-4],
                                            output_folder=os.path.join(cfg_output_folder, root.split(os.sep)[-2]),
                                            size_parameter=max_side_length,
                                            class_file=class_file)
            #xml_parser.scale(max_side_length)
            xml_parser.save_as_mask(pad=cfg_pad, thickness=thickness, dot_thickness=dot_thickness)