import os
import numpy as np
from Pillow import Image

FIELDS_FOLDER = 'fields'
ROADS_FOLDER = 'roads'

for f in os.listdir(FIELDS_FOLDER):
    im = Image.open(f)
    print(im.size)