from abc import ABCMeta, abstractmethod
import pickle


class Model(object, metaclass=ABCMeta):
    # def __init__(self):

    def save_model(self, model_file):
        with open(model_file, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(model_file):
        # static so can be called as Model.load_model('examplemodel.model')
        with open(model_file, "rb") as file:
            model = pickle.load(file)
        return model

    @abstractmethod
    def train(self, input_file):
        pass

    @abstractmethod
    def classify(self, input_file):
        pass
