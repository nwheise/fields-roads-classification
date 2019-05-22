import os
import numpy as np
from PIL import Image

DATA_FOLDER = 'data'
FIELDS_FOLDER = 'fields'
ROADS_FOLDER = 'roads'

min_dims = [np.inf, np.inf]
for folder in [FIELDS_FOLDER, ROADS_FOLDER]:
    folder_path = os.path.join(DATA_FOLDER, folder)
    
    for filename in os.listdir(folder_path):
        f = os.path.join(folder_path, filename)
        im = Image.open(f)

        for i in range(2):
            if im.size[i] < min_dims[i]:
                min_dims[i] = im.size[i]

print(f'Min Height: {min_dims[0]}')
print(f'Min Width: {min_dims[1]}')