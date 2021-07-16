import importlib
import os

from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Flatten
from tensorflow.python.keras.models import Sequential

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
                 optimizer=Adam(), loss='categorical_crossentropy', train_n_layers=None):
        """
        Constructor method
        :param base_model_name: Base model name
        :param weights: Weights of the model, initialized to imagenet
        """
        self.base_model = base_model
        self.weights = weights
        self.input_shape = input_shape
        self.model = None
        self.loss = loss

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
        self.base_model = base_cnn(input_shape=self.input_shape, weights=self.weights,
                                   pooling='avg', include_top=False)
        # self.model = Flatten()(self.base_model.output)
        # self.model = Dropout(0.5)(self.base_model)
        # self.model = Dense(512, activation='relu')(self.model)
        # self.model = Dense(10, activation='softmax')(self.model)
        return self.model

    def compile(self):
        """
        Compile the Model
        """
        self.model.compile(optimizer=Adam(), loss=self.loss)

    def get_preprocess_input(self):
        return getattr(self.base_module, 'preprocess_input')
