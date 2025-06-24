import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


nusc_class_frequencies = np.array([
 944004,
 1897170,
 152386,
 2391677,
 16957802,
 724139,
 189027,
 2074468,
 413451,
 2384460,
 5916653,
 175883646,
 4275424,
 51393615,
 61411620,
 105975596,
 116424404,
 1892500630
 ])



nusc_class_names = [
    "empty", # 0
    "barrier", # 1
    "bicycle", # 2 
    "bus", # 3 
    "car", # 4
    "construction", # 5
    "motorcycle", # 6
    "pedestrian", # 7
    "trafficcone", # 8
    "trailer", # 9
    "truck", # 10
    "driveable_surface", # 11
    "other", # 12
    "sidewalk", # 13
    "terrain", # 14
    "mannade", # 15 
    "vegetation", # 16
]

classname_to_color = {  # RGB.
    # 0: (0, 0, 0),  # Black. noise
    1: (112, 128, 144),  # Slategrey barrier
    2: (220, 20, 60),  # Crimson bicycle
    3: (255, 127, 80),  # Orangered bus
    4: (255, 158, 0),  # Orange car
    5: (233, 150, 70),  # Darksalmon construction
    6: (255, 61, 99),  # Red motorcycle
    7: (0, 0, 230),  # Blue pedestrian
    8: (47, 79, 79),  # Darkslategrey trafficcone
    9: (255, 140, 0),  # Darkorange trailer
    10: (255, 99, 71),  # Tomato truck
    11: (0, 207, 191),  # nuTonomy green driveable_surface
    12: (175, 0, 75),  # flat other
    13: (75, 0, 75),  # sidewalk
    14: (112, 180, 60),  # terrain
    15: (222, 184, 135),  # Burlywood mannade
    16: (0, 175, 0),  # Green vegetation
}

