"""
@authors:
Akshey Kumar
Vittorio Boarini
"""

import torch



#class ScaleInvariantMSE(tf.keras.losses.Loss):
#    def __init__(self, name="scale_invariant_mse"):
#        super().__init__(name=name)
#
#    def call(self, y_true, y_pred):
#        # Ensure numerical stability by adding a small epsilon before taking the log
#        epsilon = tf.keras.backend.epsilon()
#
#        # Compute the logarithms of the true and predicted values
#        log_y_true = tf.math.log(tf.abs(y_true) + epsilon)
#        log_y_pred = tf.math.log(tf.abs(y_pred) + epsilon)
#
#        # Compute the first term: mean squared error in log space
#        log_mse = tf.reduce_mean(tf.square(log_y_pred - log_y_true), axis=-1)
#
#        # Compute the second term: squared mean of log differences
#        log_diff_mean = tf.reduce_mean(log_y_pred - log_y_true, axis=-1)
#        log_diff_mean_sq = tf.square(log_diff_mean)
#
#        # Compute the scale-invariant MSE
#        scale_invariant_mse = log_mse #- log_diff_mean_sq
#
#        return scale_invariant_mse



class BccDccLoss:
    """Calculate the loss for the BunDLe Net

    Args:
        yt1_upper: Output from the upper arm of the BunDLe Net.
        yt1_lower: Output from the lower arm of the BunDLe Net.
        bt1_upper: Predicted output from the upper arm of the BunDLe Net.
        b_train_1: True output for training.
        gamma (float): Tunable weight for the DCC loss component.

    Returns:
        tuple: A tuple containing the DCC loss, behavior loss, and total loss.
    """

    def __init__(self, b_type, gamma):
        self.b_type = b_type
        self.gamma = gamma

        if b_type == 'discrete':
            self.loss_functions = {
                'd_loss_func': torch.nn.MSELoss(),
                'b_loss_func': torch.nn.CrossEntropyLoss()
            }
        elif b_type == 'continuous':
            self.loss_functions = {
                'd_loss_func': torch.nn.MSELoss(),
                'b_loss_func': torch.nn.MSELoss()
            }
        else:
            raise ValueError('Unknown loss type')

    def __call__(self, yt1_upper, yt1_lower, bt1_upper, b_train_1):
        DCC_loss = self.loss_functions['d_loss_func'](yt1_upper, yt1_lower)
        if self.b_type == 'discrete':
            behaviour_loss = self.loss_functions['b_loss_func'](bt1_upper, b_train_1.long())
        else:
            behaviour_loss = self.loss_functions['b_loss_func'](b_train_1.float(), bt1_upper)
        total_loss = self.gamma * DCC_loss + (1 - self.gamma) * behaviour_loss
        return self.gamma * DCC_loss, (1 - self.gamma) * behaviour_loss, total_loss


# @tf.function
# def contrastive_loss(y1, y2, b1, b2, margin=1.0):
#     """Computes the contrastive loss between `y_true` and `y_pred`.
#
#     Args:
#       y1:
#       y2:
#       b1: label
#       b2: label
#       margin: margin term in the loss definition.
#
#     Returns:
#       contrastive_loss: 1-D float `Tensor` with shape `[batch_size]`.
#     """
#     y1 = tf.convert_to_tensor(y1)
#     y2 = tf.convert_to_tensor(y2)
#     d = tf.norm(y1 - y2)
#
#     if b1 == b2:
#         loss = tf.math.square(d)
#     else:
#         loss = tf.math.square(0.0, tf.math.maximum(margin - d))
#     # return loss
#
#     raise NotImplementedError("This function is not yet implemented.")
