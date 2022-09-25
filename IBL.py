import IanClass
import EthanClass


class IBL (IanClass, EthanClass):
    def __init__(self, file, features, name, classLoc, replaceValue = None):
        IanClass.__init__(self, file, features, name, classLoc, replaceValue = None)