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


####################################################################
#. Load Model
setup_logger()
#create detectron config
cfg = get_cfg()
#set yaml
cfg.merge_from_file('weights/All_X152.yaml')
#set model weights
cfg.MODEL.WEIGHTS = 'weights/model_final.pth' # Set path model .pth
predictor = DefaultPredictor(cfg)

