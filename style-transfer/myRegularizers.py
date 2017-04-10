from keras import backend as K
from keras.regularizers import Regularizer


# the gram matrix of an image tensor (feature-wise outer product)
# reshape output to change th format to tf format
def gram_matrix(x):
    assert K.ndim(x) == 4
    xs = K.shape(x)
    # features = K.reshape(x, (xs[0], xs[1], xs[2] * xs[3]))
    # gram = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1)))
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    features = K.reshape(x, (xs[0], xs[1], xs[2] * xs[3]))
    gram = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1)))
    return gram


class FeatureStyleRegularizer(Regularizer):
    '''Gatys et al 2015 http://arxiv.org/pdf/1508.06576.pdf'''

    def __init__(self, target=None, weight=1.0, **kwargs):
        self.target = target
        self.weight = weight
        super(FeatureStyleRegularizer, self).__init__(**kwargs)

    def __call__(self, loss):
        output = self.layer.get_output(True)
        batch_size = K.shape(output)[0] // 2
        generated = output[:batch_size, :, :, :]
        loss += self.weight * K.mean(
            K.sum(K.square(gram_matrix(self.target) - gram_matrix(generated)), axis=(1, 2))
        ) / (4.0 * K.square(K.prod(K.shape(generated)[1:])))
        return loss


class FeatureContentRegularizer(Regularizer):
    '''Penalizes euclidean distance of content features.'''

    def __init__(self, weight=1.0, **kwargs):
        self.weight = weight
        super(FeatureContentRegularizer, self).__init__(**kwargs)

    def __call__(self, loss):
        output = self.layer.get_output(True)
        batch_size = K.shape(output)[0] // 2
        generated = output[:batch_size, :, :, :]
        content = output[batch_size:, :, :, :]
        loss += self.weight * K.mean(
            K.sum(K.square(content - generated), axis=(1, 2, 3))
        )
        return loss


# reshape output to change th format to tf format
class TVRegularizer(Regularizer):
    '''Enforces smoothness in image output.'''

    def __init__(self, weight=1.0, **kwargs):
        self.weight = weight
        super(TVRegularizer, self).__init__(**kwargs)

    def __call__(self, loss):
        x = self.layer.get_output(True)
        assert K.ndim(x) == 4
        # a = K.square(x[:, :, 1:, :-1] - x[:, :, :-1, :-1])
        # b = K.square(x[:, :, :-1, 1:] - x[:, :, :-1, :-1])
        a = K.square(x[:, 1:, :-1, :] - x[:, :-1, :-1, :])
        b = K.square(x[:, :-1, 1:, :] - x[:, :-1, :-1, :])
        loss += self.weight * K.mean(K.sum(K.pow(a + b, 1.25), axis=(1, 2, 3)))
        return loss
