"""
@authors:
Akshey Kumar
"""

import tensorflow as tf


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
                'd_loss_func': tf.keras.losses.MeanSquaredError(),
                'b_loss_func': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            }
        elif b_type == 'continuous':
            self.loss_functions = {
                    'd_loss_func': tf.keras.losses.MeanSquaredError(),
                    'b_loss_func': tf.keras.losses.MeanSquaredError()
                }

    @tf.function
    def __call__(self, yt1_upper, yt1_lower, bt1_upper, b_train_1):
        DCC_loss = self.loss_functions['d_loss_func'](yt1_upper, yt1_lower)
        behaviour_loss = self.loss_functions['b_loss_func'](b_train_1, bt1_upper)
        total_loss = self.gamma * DCC_loss + (1 - self.gamma) * behaviour_loss
        return self.gamma*DCC_loss, (1-self.gamma)*behaviour_loss, total_loss



@tf.function
def contrastive_loss(y1, y2, b1, b2, margin = 1.0):
    """Computes the contrastive loss between `y_true` and `y_pred`.

    Args:
      y1:
      y2:
      b1: label
      b2: label
      margin: margin term in the loss definition.

    Returns:
      contrastive_loss: 1-D float `Tensor` with shape `[batch_size]`.
    """
    y1 = tf.convert_to_tensor(y1)
    y2 = tf.convert_to_tensor(y2)
    d = tf.norm(y1-y2)

    #tf.math.square(d) if b1=b2
    #tf.math.square(0.0, tf.math.maximum(margin - d)) if b1 !=b2
    
    raise NotImplementedError("This function is not yet implemented.")




