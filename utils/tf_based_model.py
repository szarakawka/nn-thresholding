

class TFBasedModel:

    def train(self, train_data, val_data):
        return NotImplementedError("Please implement this method")

    def predict(self, test_data):
        return NotImplementedError("Please implement this method")

    def save_model(self):
        return NotImplementedError("Please implement this method")

    @staticmethod
    def load_model():
        return NotImplementedError("Please implement this method")
