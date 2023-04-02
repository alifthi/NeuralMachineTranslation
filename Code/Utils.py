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
        engCorpusLen = 47
        spCourpusLen = 49 + 2
        engVocabSize = 1000
        spVocabSize = 1500
        self.spanishTokenizer = tf.keras.layers.TextVectorization(max_tokens= spVocabSize,
                                                             output_sequence_length=spCourpusLen,
                                                             standardize=self.customStd)
        self.englishTokenizer = tf.keras.layers.TextVectorization(output_sequence_length=engCorpusLen,
                                                             max_tokens=engVocabSize,
                                                             standardize=self.customStd)
        self.englishTokenizer.adapt(eng)
        self.spanishTokenizer.adapt(spanish)
        return [spanish,eng]
    def vectorization(self,spanishDS,englishDS):
        spanishDSInput = [corp[:-5] for corp in spanishDS]
        spanishDSOutput = [corp[7:] for corp in spanishDS]
        englishVectors = self.englishTokenizer(englishDS)
        spanishVectorsInputs = self.spanishTokenizer(spanishDSInput)
        spanishVectorsOutput = self.spanishTokenizer(spanishDSOutput)
        return [englishVectors,spanishVectorsInputs,spanishVectorsOutput]
    @staticmethod
    def customStd(txt):
        puncs = string.punctuation
        puncs = puncs.replace(']','')
        puncs = puncs.replace('[','')    
        txt = tf.strings.lower(txt)
        return tf.strings.regex_replace(txt, f"[{re.escape(puncs)}]", "")