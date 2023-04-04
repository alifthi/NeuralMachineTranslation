from Utils import Data
from Model import Model
engVocabSize = 1000
spVocabSize = 1500
data = Data(r"C:\Users\alifa\Documents\AI\robotech\transformers\docs\w4\nmt example\nmt example\spa.txt")
data.loadData()
data.dataProperties()
spanihsCorpus,englishCorpus = data.preprocessing(engVocabSize=engVocabSize,spVocabSize=spVocabSize)
engVector,spInpVector,spOutVector =  data.vectorization(spanihsCorpus,englishCorpus)
model = Model(numEnglishTokens=engVocabSize,numSpanishTokens=spVocabSize)
model.buildModel()
model.compileModel()
model.trainModel([engVector,spInpVector,spOutVector],epochs = 5)
model.saveModel(r'C:\Users\alifa\Documents\AI\Git\english-spnaish Translator\Model')