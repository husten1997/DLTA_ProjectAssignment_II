import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Src.DataPreprocessing import generateEncodingMatrix, dataEncoding, dataDecoder
from Src.prompt import prompt


def classifierNN(sampleTrain: pd.DataFrame, sampleTest: pd.DataFrame, docRepresentation: pd.DataFrame,  data: pd.DataFrame, encoding_matrix: pd.DataFrame, label_col: str ='label_l1',
                 epochs: int =20, prediction_suffix: str ="_NNpred", prob_suffix: str = "_NNprob", batch_size = 128, project_name = "NNClassifier", plot_title = "") -> pd.DataFrame:
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
    prompt("Starting NN Classifier")
    label_col_enc = label_col + "_enc"
    sampleTrain[label_col_enc] = dataEncoding(sampleTrain['labels'], encoding_matrix)
    sampleTest[label_col_enc] = dataEncoding(sampleTest['labels'], encoding_matrix)
    #data[label_col_enc] = dataEncoding(data[label_col], encoding_matrix)
    data_ = pd.DataFrame()
    data_[label_col] = data
    data_[label_col_enc] = dataEncoding(data, encoding_matrix)

    unlist = lambda x: np.array([np.array(i) for i in np.array(x)])

    docTrain = unlist(sampleTrain['doc'])
    docTest = unlist(sampleTest['doc'])
    docRepresentation = unlist(docRepresentation)

    model = fitNN(docTrain, docTest, sampleTrain[label_col_enc], sampleTest[label_col_enc], epochs=epochs, batch_size=batch_size, project_name=project_name, plot_title = plot_title)

    pred_data = model.predict(docRepresentation.astype('float32'))
    pred_data = np.array(pred_data).reshape(-1)
    #print(pred_data)
    pred_data_groups = np.floor(pred_data + 0.5).astype('int')
    #print(pred_data)
    data_[str(label_col + prediction_suffix)] = dataDecoder(pred_data_groups, encoding_matrix)
    data_[str(label_col + prob_suffix)] = pred_data

    return data_


def fitNN(docRepTrain: pd.DataFrame, docRepTest: pd.DataFrame, labelsTrain: pd.Series, labelsTest: pd.Series, epochs: int=20, batch_size = 128, project_name = "NNClassifier", plot_title = ""):
    """
    Handles the fitting of the keras tuner and selects the best model. This model is then trained and returned.

    :param docRepTrain: document representation of the train data
    :param docRepTest: document representation of the test data
    :param labelsTrain: series of labels of the train data
    :param labelsTest: series of labels of the test data
    :param epochs: number of epochs
    :return: tensorflow model
    """

    docRepTrain = np.matrix([np.array(i) for i in docRepTrain])
    docRepTest = np.matrix([np.array(i) for i in docRepTest])

    labelsTrain = labelsTrain.to_numpy()
    labelsTest = labelsTest.to_numpy()

    input_shape = docRepTrain.shape[1]

    def createModel(hp):
        """
        Handles the creation of the Neural Network, required for the keras tuner step.

        :param hp: Hyper parameters of keras tuner
        :return: NeuralNetwork
        """
        nonlocal input_shape
        input_layer = tf.keras.layers.Input(shape=(input_shape,))  #
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

    prompt("Starting keras tuner")
    tuner = kt.BayesianOptimization(createModel, objective='val_loss', max_trials=5, overwrite=True,
                                    project_name=project_name) #, input_shape = docRepTrain.shape[1]
    tuner.search(x=docRepTrain.astype('float32'), y=labelsTrain.reshape(-1, 1), epochs=10, #.factorize()[0]
                 validation_data=(docRepTest.astype('float32'), labelsTest.reshape(-1, 1)), #.factorize()[0]
                 batch_size=batch_size)

    classifier_model = tuner.get_best_models()[0]

    prompt("Fitting best model")
    history = classifier_model.fit(x=docRepTrain.astype('float32'), y=labelsTrain.reshape(-1, 1), #.factorize()[0]
                                   epochs=epochs,
                                   validation_data=(
                                       docRepTest.astype('float32'), labelsTest.reshape(-1, 1)), #.factorize()[0]
                                   batch_size=batch_size)

    # History Plot
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation_loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(plot_title)
    plt.legend()
    plt.show()

    return classifier_model