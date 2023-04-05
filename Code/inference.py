import tensorflow as tf
import numpy as np

modelPath = r"C:\Users\alifa\Documents\AI\Git\english-spnaish Translator\Model\model.h5"
model = tf.keras.models.load_model(modelPath)
model.summary()
encoderInput = tf.keras.layers.Input([None])
encoderEmbedding = model.layers[2](encoderInput)
encoderLstm = model.layers[4](encoderEmbedding)
encoderModel = tf.keras.Model(encoderInput,encoderLstm)

decoderInput = tf.keras.layers.Input([None])
decoderCState = tf.keras.layers.Input([None])
decoderHState = tf.keras.layers.Input([None])
decoderEmbedding = model.layers[3](decoderInput)
decoderOutput,Hstate,Cstate = model.layers[5](decoderEmbedding,initial_state =[decoderCState,decoderHState])

decoderDense = model.layers[6](decoderOutput)
decoderModel = tf.keras.Model([decoderInput,decoderCState,decoderHState],decoderDense)
def inference(inputSentence,vectorizer,decoderOutputLen):
    inputVector = vectorizer(inputSentence)
    state = encoderModel.predict(inputVector)
    decoderResponse = []
    target = np.expand_dims([41],axis=0)   # 41 is index of [Start] special token
    while True:    
        output,HState,CState = decoderModel.predict([target]+state)
        predictedWord = np.argmax(output)
        decoderResponse.append(predictedWord)
        target = np.argmax(predictedWord,axis = 0)
        state = [HState,CState]
        if decoderOutputLen < len(decoderResponse) or target == 3:
            break
    return decoderResponse
        