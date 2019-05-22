import os
import numpy as np
from PIL import Image

FIELDS_FOLDER = 'fields'
ROADS_FOLDER = 'roads'

for f in os.listdir(FIELDS_FOLDER):
    fname = os.path.join(FIELDS_FOLDER, f)
    im = Image.open(fname)
    print(im.size)
