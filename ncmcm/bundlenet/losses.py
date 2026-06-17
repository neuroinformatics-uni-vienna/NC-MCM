"""
@authors:
Akshey Kumar
Vittorio Boarini
"""

import torch


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
