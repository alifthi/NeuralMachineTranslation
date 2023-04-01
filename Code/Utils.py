import tensorflow as tf
import pandas as pd
import string
import re
class Data:
    def __init__(self,path) -> None:
        self.path = path
    def loadData(self):
        self.data = pd.read_csv(self.path,sep='\t')
        self.data.columns = ['eng','sp']
    def preprocessing(self):
        spanish = self.data['sp']
        eng = self.data['eng']
        spanish = '[Start] ' + spanish+ ' [End]'
