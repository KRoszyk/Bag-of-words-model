import os
from pathlib import Path
import cv2

data_path = Path('./../Projekt ZPO/train/uam')
for file in data_path.iterdir():
    img_file = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    if img_file is None:
        os.remove(file)
