import tensorflow as tf
from tensorflow.keras import layers as ksl
class Model:
    def __init__(self,numEnglishTokens,numSpanishTokens) -> None:
        self.numEnglishTokens = numEnglishTokens
        self.numSpanishTokens = numSpanishTokens
    def buildModel(self):
        engInp = ksl.Input([None])
        x = ksl.Embedding(input_dim = self.numEnglishTokens,output_dim = 128)(engInp)
        engGru = ksl.GRU(64)(x)
        spInp = ksl.Input(self.numSpanishTokens)
        x = ksl.Embedding(input_dim = self.numSpanishTokens,output_dim = 128)(spInp)
        x = ksl.GRU(64,return_sequences = True)(x,initial_state = [engGru])
        x = ksl.TimeDistributed(ksl.Dense(self.numSpanishTokens,'softmax'))(x)
        self.net = tf.keras.Model(inputs = [engInp,spInp],outputs = x)
    def compileModel(self):
        Adam = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.net.compile(
            optimizer=Adam,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
model = Model(100,150)
model.buildModel()
model.compileModel()
        