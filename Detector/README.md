Пример подключения детектора:

У нас есть некоторый main.py, где мы хотим использовать детектор. 
В main.py необходимо сделать следующее:

import os
import sys
from Detector import Detector

sys.path.append(os.path.abspath(os.getcwd()) + '/Detector/yolov5')

...

det = Detector.YoloDetector()
res = det.detect(img)


