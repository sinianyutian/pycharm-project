import keras

from keras_module.data_draw.draw_pascal import draw_images_pair


class DrawEpoch(keras.callbacks.Callback):
    def __init__(self, data_test, model, period=20, draw_function=draw_images_pair):
        super(DrawEpoch, self).__init__()
        self.period = period
        self.data_test = data_test
        self.model = model
        self.current_epoch = 0
        self.draw_function = draw_function

    def on_epoch_end(self, epoch, logs=None):
        data_label = self.model.predict(self.data_test)
        if self.current_epoch % self.period == self.period - 1:
            self.draw_function(self.data_test, data_label, epoch)
            print("Saving {0} img.".format(epoch))
            print("   ")

        self.current_epoch += 1

