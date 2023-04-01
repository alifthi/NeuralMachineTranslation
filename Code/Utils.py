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
    def dataProperties(self):
        dataWordCount = []
        for j in ['eng','sp']:
            lenOfWords = []
            for i in self.data[j]:
                lenOfWords.append(len(i.split(' ')))
            dataWordCount.append(lenOfWords)
        lenOfWords = pd.DataFrame(dataWordCount)
        lenOfWords = lenOfWords.T
        lenOfWords.columns = ['eng','sp']
        print(lenOfWords.describe())

    def preprocessing(self):
        spanish = self.data['sp']
        eng = self.data['eng']
        spanish = '[Start] ' + spanish+ ' [End]'
data = Data(r"C:\Users\alifa\Documents\AI\robotech\transformers\docs\w4\nmt example\nmt example\spa.txt")
data.loadData()
data.dataProperties()
data.preprocessing()
           