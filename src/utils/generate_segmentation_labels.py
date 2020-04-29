import xml.etree.ElementTree as ET
import os
import cv2
import math
import copy
import numpy as np
from src.utils.point import Point
from src.utils.utils import load_class_dict


class XMLParserBaselines:
    """
    Reads a Page XML file and generates the corresponding label files that contain of a mask for the baselines.
    The colors are given in self.colors. Also allows resizing.
    """
    def __init__(self, xml_filename: str, input_folder: str, output_folder: str, size_parameter: int, class_file: str):
        # Load the XML file
        self.tree = ET.parse(os.path.join(xml_filename))
        self.root = self.tree.getroot()

        # Extract name and size data
        self.filename = self.root.getchildren()[1].attrib['imageFilename']
        self.width = int(self.root.getchildren()[1].attrib['imageWidth'])
        self.height = int(self.root.getchildren()[1].attrib['imageHeight'])

        self.input_folder = input_folder
        self.output_folder = output_folder

        self.size_parameter = size_parameter #change, ugly
        self.region_types, _, self.colors = load_class_dict(class_file)

        self.scaled = False

        self.baselines, _ = self.extract_points()
        self.scale_baselines(self.size_parameter)

        if size_parameter == 1024:
            diff = 13#15
            height = 8#10
        elif size_parameter == 512:
            diff = 10
            height = 7
        else:
            raise NotImplementedError

        self.start_points = self.extract_start_points(self.baselines)
        self.end_points = self.extract_end_points(self.baselines)
        self.seperators, self.angles_start, self.angles_end = self.extract_seperators(self.baselines, diff=diff)
        self.baseline_border = copy.deepcopy(self.baselines)
        self.text = self.create_text_mask(self.baselines, self.angles_start, self.angles_end, height=height)
        self.regions = {'baseline_border': self.baseline_border,
                        'text': self.text,
                        'baselines': self.baselines,
                        'seperators': self.seperators,
                        'start_points': self.start_points,
                        'end_points': self.end_points}

    def extract_points(self) -> list:
        """
        Extracts the text regions of the xml file as polygons and saves them in a dict.
        :return:    The dict where for every region type a list of polygons (that are again list of Points)
                    is given. The polygons are exactly the text regions described in the Page XML file.
        """
        baselines = []
        baseline_regions = []

        for region in self.root.getchildren()[1]:
            if 'TextRegion' in region.tag:
                for child in region:
                    if 'TextLine' in child.tag:
                        if len(child.getchildren()) > 1:
                            # Region
                            coord_string_region = child.getchildren()[0].attrib['points']
                            points_string_region = coord_string_region.split()
                            points_region = []

                            for p in points_string_region:
                                point_region = Point()
                                point_region.set_from_string(coords=p, sep=',')
                                points_region.append(point_region)

                            # Baseline
                            coord_string_line = child.getchildren()[1].attrib['points']
                            points_string_line = coord_string_line.split()
                            points_line = []

                            for p in points_string_line:
                                point_line = Point()
                                point_line.set_from_string(coords=p, sep=',')
                                points_line.append(point_line)

                            if len(points_line) > 1 and len(points_region) > 1:
                                baselines.append(points_line)
                                baseline_regions.append(points_region)

        return baselines, baseline_regions

    @staticmethod
    def extract_start_points(baselines) -> list:
        start_points = [copy.deepcopy(bl[0]) for bl in baselines]
        return start_points

    @staticmethod
    def extract_end_points(baselines) -> list:
        end_points = [copy.deepcopy(bl[-1]) for bl in baselines]
        return end_points

    @staticmethod
    def extract_seperators(baselines, diff) -> list:
        start_points = [bl[0] for bl in baselines]
        angle_list_start = []
        angle_list_end = []

        seperators = []

        for bl in baselines:
            # diff = self.compute_text_height(bl, self.text)
            bl_list = np.array([p.get_as_list() for p in bl])

            idx = 0
            if np.abs(bl_list[idx, 0] - bl_list[idx + 1, 0]) < 0.001:
                if bl_list[idx, 1] > bl_list[idx + 1, 1]:
                    angle = math.pi / 2.0
                else:
                    angle = -math.pi / 2.0
            else:
                if bl_list[idx, 0] < bl_list[idx + 1, 0]:
                    angle = np.arctan((bl_list[idx + 1, 1] - bl_list[idx, 1]) / (bl_list[idx, 0] - bl_list[idx + 1, 0]))
                else:
                    if np.abs(bl_list[idx, 1] - bl_list[idx + 1, 1]) < 0.001:
                        if bl_list[idx, 1] > bl_list[idx + 1, 1]:
                            angle = math.pi/2
                        else:
                            angle = -math.pi/2
                    else:
                        if bl_list[idx, 1] > bl_list[idx + 1, 1]:
                            angle = np.arctan(
                                (bl_list[idx, 0] - bl_list[idx + 1, 0]) / (bl_list[idx, 1] - bl_list[idx + 1, 1])) + math.pi/2
                        else:
                            angle = np.arctan(
                                (bl_list[idx, 0] - bl_list[idx + 1, 0]) / (bl_list[idx, 1] - bl_list[idx + 1, 1])) - math.pi/2

            p0 = copy.deepcopy(bl[0])

            x1 = p0.x - diff/1.5*np.cos(angle + math.pi/2)
            y1 = p0.y + diff/1.5*np.sin(angle + math.pi/2)
            p1 = Point(x1, y1)

            x2 = p0.x + diff*np.cos(angle + math.pi/2)
            y2 = p0.y - diff*np.sin(angle + math.pi/2)
            p2 = Point(x2, y2)

            seperators.append([p1, p2])

            angle_list_start.append(angle)

        for bl in baselines:
            # diff = self.compute_text_height(bl, self.text)
            bl_list = np.array([p.get_as_list() for p in bl])

            idx = -2
            if np.abs(bl_list[idx, 0] - bl_list[idx + 1, 0]) < 0.001:
                if bl_list[idx, 1] > bl_list[idx + 1, 1]:
                    angle = math.pi / 2.0
                else:
                    angle = -math.pi / 2.0
            else:
                if bl_list[idx, 0] < bl_list[idx + 1, 0]:
                    angle = np.arctan((bl_list[idx + 1, 1] - bl_list[idx, 1]) / (bl_list[idx, 0] - bl_list[idx + 1, 0]))
                else:
                    if np.abs(bl_list[idx, 1] - bl_list[idx + 1, 1]) < 0.001:
                        if bl_list[idx, 1] > bl_list[idx + 1, 1]:
                            angle = math.pi/2
                        else:
                            angle = -math.pi/2
                    else:
                        if bl_list[idx, 1] > bl_list[idx + 1, 1]:
                            angle = np.arctan(
                                (bl_list[idx, 0] - bl_list[idx + 1, 0]) / (bl_list[idx, 1] - bl_list[idx + 1, 1])) + math.pi/2
                        else:
                            angle = np.arctan(
                                (bl_list[idx, 0] - bl_list[idx + 1, 0]) / (bl_list[idx, 1] - bl_list[idx + 1, 1])) - math.pi/2

            p0 = copy.deepcopy(bl[-1])

            x1 = p0.x - diff/1.5*np.cos(angle + math.pi/2)
            y1 = p0.y + diff/1.5*np.sin(angle + math.pi/2)
            p1 = Point(x1, y1)

            x2 = p0.x + diff*np.cos(angle + math.pi/2)
            y2 = p0.y - diff*np.sin(angle + math.pi/2)
            p2 = Point(x2, y2)

            seperators.append([p1, p2])

            angle_list_end.append(angle)

        return seperators, angle_list_start, angle_list_end

    @staticmethod
    def create_text_mask(baselines, angles_start, angles_end, height):
        text_regions = copy.deepcopy(baselines)

        for n, bl in enumerate(text_regions):
            bl_tmp = copy.deepcopy(bl)
            for k, p in enumerate(bl_tmp[::-1]):
                if k == 0:
                    angle = angles_end[n]
                else:
                    angle = angles_start[n]

                x = p.x + height * np.cos(angle + math.pi / 2)
                y = p.y - height * np.sin(angle + math.pi / 2)
                top_point = Point(int(x), int(y))

                bl.append(top_point)

        return text_regions

    def save_as_mask(self, pad: bool=False, thickness=3, dot_thickness=2):
        """
        Generate a mask with the extracted polygons drawn according to the colors self.colors.
        :return: The generated label image as numpy array
        """
        if not os.path.isdir(os.path.join(self.output_folder, 'images')):
            os.mkdir(os.path.join(self.output_folder, 'images'))
        if not os.path.isdir(os.path.join(self.output_folder, 'labels')):
            os.mkdir(os.path.join(self.output_folder, 'labels'))

        if pad:
            max_side = max(self.height, self.width)
            img = np.zeros((max_side, max_side, 3), np.uint8)
        else:
            img = np.zeros((self.height, self.width, 3), np.uint8)

        for region_type in self.region_types:
            if region_type in ['start_points', 'end_points']:
                region = self.regions[region_type]
                for p in region:
                    cv2.circle(img=img, center=(p.x, p.y), radius=dot_thickness+1, color=self.colors[region_type],
                               thickness=-1)
            elif region_type == 'sp_ep_border':
                region = self.regions['start_points']
                for p in region:
                    cv2.circle(img=img, center=(p.x, p.y), radius=int(1.5*dot_thickness)+2, color=self.colors[region_type],
                               thickness=-1)
                region = self.regions['end_points']
                for p in region:
                    cv2.circle(img=img, center=(p.x, p.y), radius=int(1.5*dot_thickness)+2, color=self.colors[region_type],
                               thickness=-1)
            elif region_type == 'bg':
                continue
            else:
                for region in self.regions[region_type]:
                    points_array = []
                    for p in region:
                        points_array.append(p.get_as_list())
                    pts = np.array(points_array, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    if region_type == 'baseline_border':
                        cv2.polylines(img=img, pts=[pts], isClosed=False, color=self.colors[region_type],
                                      thickness=(thickness*3))
                    if region_type == 'baselines':
                        cv2.polylines(img=img, pts=[pts], isClosed=False, color=self.colors[region_type],
                                      thickness=thickness)
                    elif region_type == 'text':
                        cv2.fillPoly(img=img, pts=[pts], color=self.colors[region_type])
                    else:
                        cv2.polylines(img=img, pts=[pts], isClosed=False, color=self.colors[region_type],
                                      thickness=thickness)

        cv2.imwrite(os.path.join(self.output_folder, 'labels', self.filename.split('.')[0] + '.png'), img)

        if self.scaled:
            scan = cv2.imread(os.path.join(self.input_folder, self.filename))
            scan_resized = cv2.resize(scan, (self.width, self.height))
            if pad:
                scan_resized = np.pad(scan_resized, ((0, max_side-self.height), (max_side-self.width, 0), (0,0)),
                                      mode='constant', constant_values=0)

            cv2.imwrite(os.path.join(self.output_folder, 'images', self.filename.split('.')[0] + '.png'), scan_resized)

        return img

    def scale_baselines(self, min_side: int):
        """
        For images with at least one side larger than max_side scales the image and polygons
        such that the maximal side length is given by max_side.
        If max_side is larger than the maximum of width and height nothing is done.
        :param max_side: Maximally allowed side length
        """
        if self.scaled:
            return
        else:
            ratio = min_side / min(self.width, self.height)
            w = self.width
            h = self.height

            self.width = min_side if w < h else round(w * ratio)
            self.height = min_side if h < w else round(h * ratio)

            for region in self.baselines:
                for point in region:
                    point.scale(ratio)

            self.scaled = True

