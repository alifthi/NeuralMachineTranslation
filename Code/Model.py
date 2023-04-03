import tensorflow as tf
from tensorflow.keras import layers as ksl
class Model:
    def __init__(self,numEnglishTokens,numSpanishTokens) -> None:
        self.numEnglishTokens = numEnglishTokens
        self.numSpanishTokens = numSpanishTokens
    def buildModel(self):
        engInp = ksl.Input([None])
        x = ksl.Embedding(input_dim = self.numEnglishTokens,output_dim = 64)(engInp)
        engGru = ksl.GRU(64)(x)
        spInp = ksl.Input([None])
        x = ksl.Embedding(input_dim = self.numSpanishTokens,output_dim = 32)(spInp)
        x = ksl.GRU(64,return_sequences = True)(x,initial_state = [engGru])
        x = ksl.TimeDistributed(ksl.Dense(self.numSpanishTokens,'softmax'))(x)
        self.net = tf.keras.Model(inputs = [engInp,spInp],outputs = x)
    def compileModel(self):
        Adam = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.net.compile(
            optimizer=Adam,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
    def trainModel(self,trainData,validationSplit = 0.2,batchSize = 32,epochs = 10):
        self.net.fit(trainData[:-1],
                     trainData[-1],
                     batch_size=batchSize,
                     epochs=epochs,
                     validation_split= validationSplit)
    def saveModel(self,path):
        self.net.save(path + '\\model.h5')
    def loadModel(self,path):
        self.net = tf.keras.models.load_model(path)