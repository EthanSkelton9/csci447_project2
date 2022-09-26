import pandas as pd
import os
import math
import random
from ConfusionMatrix import ConfusionMatrix
import numpy as np
from Learning import Learning

class EthanClass (Learning):
    def __init__(self, file, features, name, classLoc, replaceValue=None, classification=True):
        super().__init__(file=file, features=features, name=name, classLoc=classLoc, replaceValue=replaceValue,
                         classification=classification)