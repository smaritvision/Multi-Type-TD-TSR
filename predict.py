import detectron2
import google_colab.deskew as deskew
import google_colab.table_detection as table_detection
import google_colab.table_structure_recognition_all as tsra
import google_colab.table_structure_recognition_lines as tsrl
import google_colab.table_structure_recognition_wol as tsrwol
import google_colab.table_structure_recognition_lines_wol as tsrlwol
import google_colab.table_xml as txml
import google_colab.table_ocr as tocr
import pandas as pd
import os
import json
import itertools
import random
from detectron2.utils.logger import setup_logger
# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog

from google_colab.plot_show import cv2_imshow

setup_logger()


#create detectron config
cfg = get_cfg()

#set yaml
cfg.merge_from_file('weights/All_X152.yaml')

#set model weights
cfg.MODEL.WEIGHTS = 'weights/model_final.pth' # Set path model .pth

predictor = DefaultPredictor(cfg)


# path to the image scan of the document
file = "images/rotated_example.jpeg"

# load the image from disk
original_image = cv2.imread(file)
deskewed_image = deskew.deskewImage(original_image)

print("ORIGINAL IMAGE:")
cv2_imshow(original_image)
print("DESKEWED IMAGE:")
cv2_imshow(deskewed_image)


document_img = cv2.imread("images/color_invariance_example.jpg")
table_detection.plot_prediction(document_img, predictor)

table_list, table_coords = table_detection.make_prediction(document_img, predictor)

list_table_boxes = []

for table in table_list:
    finalboxes, output_img = tsra.recognize_structure(table)
    list_table_boxes.append(finalboxes)


txml.output_to_xml(table_coords, list_table_boxes)

bordered_table = cv2.imread("images/bordered_example.png")
list_table_boxes = []

table_list = [bordered_table]

for table in table_list:
    finalboxes, output_img = tsrl.recognize_structure(table)
    list_table_boxes.append(finalboxes)


#Table Structure Recognition Unbordered Tables

unbordered_table = cv2.imread("images/unbordered_example.jpg")
list_table_boxes = []

table_list = [unbordered_table]

for table in table_list:
    finalboxes, output_img = tsrwol.recognize_structure(table)
    list_table_boxes.append(finalboxes)

#Table Structure Recognition Partially Bordered Tables
document_example = cv2.imread("images/example.jpg")

table_list, table_coords = table_detection.make_prediction(document_example, predictor)
list_table_boxes = []
for table in table_list:
    finalboxes, output_img = tsrlwol.recognize_structure(table)
    list_table_boxes.append(finalboxes)

#Additional Feature: OCR with PyTesserect (Output in CSV)
document_example = cv2.imread("images/example.jpg")

table_list, table_coords = table_detection.make_prediction(document_example, predictor)
list_table_boxes = []
for table in table_list:
    finalboxes, output_img = tsrlwol.recognize_structure(table)
    list_table_boxes.append(finalboxes)