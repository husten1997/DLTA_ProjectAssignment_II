import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Src.DataPreprocessing import generateEncodingMatrix, dataEncoding, dataDecoder


def createModel(hp):
    """
    Handles the creation of the Neural Network, required for the keras tuner step.

    :param hp: Hyper parameters of keras tuner
    :return: NeuralNetwork
    """
    input_layer = tf.keras.layers.Input(shape=(300,))
    output_layer = input_layer
    output_layer = tf.keras.layers.Dense(hp.Choice('L1_Units', [100, 200, 300]),
                                         activation=hp.Choice('L1_Act', ["relu", "tanh", "linear"]))(output_layer)
    output_layer = tf.keras.layers.Dense(hp.Choice('L2_Units', [50, 100, 150]),
                                         activation=hp.Choice('L2_Act', ["relu", "tanh", "linear"]))(output_layer)
    output_layer = tf.keras.layers.Dense(hp.Choice('L3_Units', [25, 50, 75]),
                                         activation=hp.Choice('L3_Act', ["relu", "tanh", "linear"]))(output_layer)
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(output_layer)

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    return model


def classifierNN(docRepTrain: pd.DataFrame, docRepTest: pd.DataFrame, docRepresentation: pd.DataFrame, dataTrain: pd.DataFrame, dataTest: pd.DataFrame, data: pd.DataFrame, encoding_matrix: pd.DataFrame, label_col: str ='label_l1',
                 epochs: int =20, prediction_suffix: str ="_NNpred", prob_suffix: str = "_NNprob") -> pd.DataFrame:
    """
    Wrapper function for the whole classification process by neural network.

    :param docRepTrain: document representation for the train dataset
    :param docRepTest: document representation for the test dataset
    :param docRepresentation: complete document representation (can be different from fitting data, i.e. all documents can be used)
    :param dataTrain: train data
    :param dataTest: test data
    :param data: complete dataset, can be different from the fitting data
    :param encoding_matrix: a metrix which determines the encoding of the labels (see Src.DataPreprocessing.generateEncodingMatrix)
    :param label_col: name of the target colum
    :param epochs: number of epochs
    :param prediction_suffix: suffix for the prediction colum
    :return: dataframe with the original data plus the colum of predicted label data
    """
    label_col_enc = label_col + "_enc"
    dataTrain[label_col_enc] = dataEncoding(dataTrain[label_col], encoding_matrix)
    dataTest[label_col_enc] = dataEncoding(dataTest[label_col], encoding_matrix)
    data[label_col_enc] = dataEncoding(data[label_col], encoding_matrix)

    model = fitNN(docRepTrain, docRepTest, dataTrain[label_col_enc], dataTest[label_col_enc], epochs=epochs)

    pred_data = model.predict(np.matrix(docRepresentation).astype('float32'))
    pred_data = np.array(pred_data).reshape(-1)
    #print(pred_data)
    pred_data_groups = np.floor(pred_data + 0.5).astype('int')
    #print(pred_data)
    data[str(label_col + prediction_suffix)] = dataDecoder(pred_data_groups, encoding_matrix)
    data[str(label_col + prob_suffix)] = pred_data

    return data


def fitNN(docRepTrain: pd.DataFrame, docRepTest: pd.DataFrame, labelsTrain: pd.Series, labelsTest: pd.Series, epochs: int=20):
    """
    Handles the fitting of the keras tuner and selects the best model. This model is then trained and returned.

    :param docRepTrain: document representation of the train data
    :param docRepTest: document representation of the test data
    :param labelsTrain: series of labels of the train data
    :param labelsTest: series of labels of the test data
    :param epochs: number of epochs
    :return: tensorflow model
    """
    tuner = kt.BayesianOptimization(createModel, objective='val_loss', max_trials=5, overwrite=True,
                                    project_name="classifierNN")
    tuner.search(x=np.matrix(docRepTrain).astype('float32'), y=labelsTrain.factorize()[0].reshape(-1, 1), epochs=10,
                 validation_data=(np.matrix(docRepTest).astype('float32'), labelsTest.factorize()[0].reshape(-1, 1)),
                 batch_size=1024)

    classifier_model = tuner.get_best_models()[0]

    history = classifier_model.fit(x=np.matrix(docRepTrain).astype('float32'), y=labelsTrain.factorize()[0].reshape(-1, 1),
                                   epochs=epochs,
                                   validation_data=(
                                       np.matrix(docRepTest).astype('float32'), labelsTest.factorize()[0].reshape(-1, 1)),
                                   batch_size=1024)

    # History Plot
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation_loss')
    plt.legend()
    plt.show()

    return classifier_model