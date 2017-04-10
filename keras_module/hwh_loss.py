import numpy as np
from keras import backend as K


def fcn32_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_pred, y_true, from_logits=True)


def fcn32_acc(y_true, y_pred):
    return K.mean(K.cast(K.equal(K.argmax(y_true, axis=-1),
                                 K.argmax(y_pred, axis=-1)),
                         K.floatx()))


def gram_matrix(x):
    assert K.ndim(x) == 4
    xs = K.shape(x)
    # features = K.reshape(x, (xs[0], xs[1], xs[2] * xs[3]))
    # gram = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1)))
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    # print(K.get_variable_shape(x))
    features = K.reshape(x, (xs[0], xs[1], xs[2] * xs[3]))
    gram = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1)))
    # print(K.get_variable_shape(gram))
    return gram


def content_loss(content, combination):
    return K.mean(
        K.sum(K.square(content - combination), axis=(1, 2, 3))
    )


def style_loss(style, combination):
    assert K.ndim(style) == 4
    assert K.ndim(combination) == 4
    target = style
    generated = combination
    var_shape = K.get_variable_shape(style)
    var_squar_prod = np.square(np.prod(var_shape[1:]))
    # print(var_squar_prod)
    return K.mean(
        K.sum(K.square(gram_matrix(target) - gram_matrix(generated)), axis=(1, 2))
    ) / (4.0 * var_squar_prod)


def tv_loss(x):
    assert K.ndim(x) == 4
    # a = K.square(x[:, :, 1:, :-1] - x[:, :, :-1, :-1])
    # b = K.square(x[:, :, :-1, 1:] - x[:, :, :-1, :-1])
    a = K.square(x[:, 1:, :-1, :] - x[:, :-1, :-1, :])
    b = K.square(x[:, :-1, 1:, :] - x[:, :-1, :-1, :])
    return K.mean(K.sum(K.pow(a + b, 1.25), axis=(1, 2, 3)))

    #
    # # y_true:content_image y_pred:generated image
    # def style_total_loss(y_true, y_pred):
    #     loss = K.variable(0.)
    #     style_reference_image = K.variable(style_image)
    #     input_tensor = K.concatenate([y_pred, y_true,
    #                                   style_reference_image], axis=0)
    #     # print(K.get_variable_shape(y_true))
    #     # print(K.get_variable_shape(input_tensor))
    #     # print(K.get_variable_shape(input_tensor))
    #     content, style_layers = vgg16_model.get_layer(input_tensor)
    #
    #     generated = content[:batch_size, :, :, :]
    #     contented = content[batch_size:2 * batch_size, :, :, :]
    #     # print(K.get_variable_shape(generated))
    #     # print(K.get_variable_shape(contented))
    #     # exit()
    #
    #     loss += weigths['loss_weight'] * K.mean(
    #         K.sum(K.square(contented - generated), axis=(1, 2, 3))
    #     )
    #
    #     for style_layer in style_layers:
    #         combination_features = style_layer[:batch_size, :, :, :]
    #         style_reference_features = style_layer[2 * batch_size:3 * batch_size, :, :, :]
    #         sl = style_loss(style_reference_features, combination_features)
    #         loss += weigths['stlye_weigth'] * sl
    #
    #     loss += weigths['tv_weigth'] * tv_loss(y_pred[:batch_size, :, :, :])
    #
    #     return loss
