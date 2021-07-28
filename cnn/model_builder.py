import importlib
import os

from pathlib import Path

import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Flatten
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import KFold
from livelossplot.inputs.keras import PlotLossesCallback

MODELS_FILE_DIR = Path(__file__).resolve().parent
MODELS_JSON_FILE_PATH = os.path.join(MODELS_FILE_DIR, 'models.json')


def get_vgg_block(num_blocks=1, input_shape=(150, 150, 3),
                  padding='same'):
    assert 4 > num_blocks > 0, f'Number of blocks should be in range 1 and 3'
    model = Sequential()
    dropout_list = [.2, .2, .3, .3]
    filter_list = [32, 64, 128, 128]

    def dropout_value(index):
        return dropout_list[index] if index <= len(dropout_list) else dropout_list[-1]

    def filter_value(index):
        return filter_list[index] if index <= len(filter_list) else filter_list[-1]

    for i in range(num_blocks):
        if i == 0:
            model.add(Conv2D(filters=filter_value(i), kernel_size=3,
                             activation='relu', input_shape=input_shape,
                             padding=padding)
                      )
        else:
            model.add(Conv2D(filters=filter_value(i),
                             kernel_size=3, activation='relu', padding=padding))
        # model.add(Conv2D(filters=filter_value(i), kernel_size=3,
        #                  activation='relu', padding=padding))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Dropout(dropout_value(i)))

    model.add(Flatten())

    return model


class CNNModel:
    def __init__(self, base_model, weights='imagenet', input_shape=(224, 224, 3),
                 optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy']):
        """
        Constructor method
        :param base_model_name: Base model name
        :param weights: Weights of the model, initialized to imagenet
        """
        self.metrics = metrics
        self.base_model = base_model
        self.weights = weights
        self.input_shape = input_shape
        self.model = None
        self.loss = loss
        self.optimizer = optimizer
        self.preprocessing_function = None

    def _get_base_module(self, model_name):
        """
        Get the base model based on the base model name
        :param model_name: Base model name
        :return: Base models' library
        """
        import json
        with open(MODELS_JSON_FILE_PATH) as model_json_file:
            models = json.load(model_json_file)
        if model_name not in models.keys():
            raise Exception(
                f"Invalid model name, should have one of the value {models.keys()}")
        self.base_model_name = models[model_name]['model_name']
        model_package = models[model_name]['model_package']
        print(f"{model_package}.{self.base_model_name}")
        self.base_module = importlib.import_module(model_package)

    def build(self):
        """
        Build the CNN model for Neural Image Assessment
        """
        # Load pre trained model
        base_cnn = getattr(self.base_module, self.base_model_name)
        self.preprocessing_function = getattr(self.base_module, 'preprocess_input')
        self.base_model = base_cnn(input_shape=self.input_shape, weights=self.weights,
                                   pooling='avg', include_top=False)
        return self.model

    def compile(self):
        """
        Compile the Model
        """
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def summary(self):
        self.model.summary()

    def get_preprocess_input(self):
        return self.preprocessing_function

    def train_model_from_dataframe(self, df, img_directory, model, x_col, y_col, monitor='val_accuracy',
                                   weight_prefix=None, weights_dir=None, class_mode='category',
                                   batch_size=32, epochs=25, verbose=0):
        train_result_df = []
        target_size = (self.input_shape[0], self.input_shape[1])
        # Take a 5 fold cross validation
        cv = KFold(n_splits=5, shuffle=True, random_state=1024)
        fold = 1

        # Loop for each fold
        for train_index, val_index in cv.split(df[x_col]):
            train_df, val_df = df.iloc[train_index], df.iloc[val_index]
            # Define Generators
            train_datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True,
                                               vertical_flip=True)
            train_gen = train_datagen.flow_from_dataframe(train_df, directory=img_directory,
                                                          x_col=x_col, y_col=y_col,
                                                          batch_size=batch_size, class_mode=class_mode,
                                                          target_size=target_size,
                                                          preprocessing_function=self.preprocessing_function)

            valid_gen = train_datagen.flow_from_dataframe(val_df, directory=img_directory,
                                                          x_col=x_col, y_col=y_col,
                                                          batch_size=batch_size, class_mode=class_mode,
                                                          target_size=target_size,
                                                          preprocessing_function=self.preprocessing_function)

            # compile model
            model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
            # Define the callbacks
            es = EarlyStopping(monitor=monitor, patience=4)

            assert (os.path.isdir(weights_dir), 'Invalid directory ' + weights_dir)
            # Assign current directory if no directory passed to save weights
            weights_dir = weights_dir if weights_dir is not None else os.getcwd()
            weight_prefix = weight_prefix if weight_prefix is not None else self.base_model_name
            weight_filepath = os.path.join(weights_dir, f'{weight_prefix}_weight_best_fold_{fold}.hdf5')
            print(f'\tModel Weight file : {weight_filepath}')
            ckpt = ModelCheckpoint(
                filepath=weight_filepath,
                save_weights_only=True,
                monitor=monitor,
                mode="max",
                save_best_only=True,
            )
            lr = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1)
            plot_loss = PlotLossesCallback()

            # start training
            history = model.fit(train_gen, validation_data=valid_gen,
                                epochs=epochs, callbacks=[es, ckpt, lr, plot_loss],
                                verbose=verbose)
            result_df = pd.DataFrame(history.history)
            result_df['fold'] = fold
            train_result_df.append(result_df)

            return pd.concat(train_result_df)
