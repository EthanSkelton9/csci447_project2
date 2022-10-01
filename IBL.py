from IanClass import IanClass
from EthanClass import EthanClass
from asyncio.windows_events import NULL
import pandas as pd
import os
import math
import random
from ConfusionMatrix import ConfusionMatrix
import numpy as np

#instance based learning
class IBL (IanClass, EthanClass):
    def __init__(self, file, features, name, classLoc,replaceValue = None, classification=True):
        super().__init__(file = file, features = features, name = name, classLoc = classLoc,
                         replaceValue = replaceValue, classification = classification)




