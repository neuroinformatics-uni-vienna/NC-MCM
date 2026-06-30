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

    def __init__(self, b_type, gamma, n_steps=1, discount=1.0, unroll='both'):
        if unroll not in ('both', 'dynamics', 'behaviour'):
            raise ValueError("unroll must be 'both', 'dynamics', or 'behaviour'")
        self.b_type = b_type
        self.gamma = gamma
        self.n_steps = n_steps
        self.unroll = unroll
        self._discount_weights = [discount ** j for j in range(n_steps)]

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

    def __call__(self, upper_ys, lower_ys, upper_bs, b_train):
        unroll = self.unroll

        # Accept legacy single-tensor interface for backward compatibility
        if isinstance(upper_ys, torch.Tensor):
            upper_ys, lower_ys, upper_bs = [upper_ys], [lower_ys], [upper_bs]

        weights = torch.tensor(self._discount_weights, dtype=torch.float, device=upper_ys[0].device)

        if unroll in ('both', 'dynamics'):
            dcc_terms = torch.stack([
                self.loss_functions['d_loss_func'](upper_ys[j], lower_ys[j])
                for j in range(self.n_steps)
            ])
            DCC_loss = (weights * dcc_terms).sum() / weights.sum()
        else:
            DCC_loss = self.loss_functions['d_loss_func'](upper_ys[0], lower_ys[0])

        if self.b_type == 'discrete':
            # b_train: (m,) at n_steps=1 or (m, n_steps) at n_steps>1
            if b_train.dim() == 1:
                b_train = b_train.unsqueeze(1)  # → (m, 1)
            if unroll in ('both', 'behaviour'):
                b_terms = torch.stack([
                    self.loss_functions['b_loss_func'](upper_bs[j], b_train[:, j].long())
                    for j in range(self.n_steps)
                ])
                behaviour_loss = (weights * b_terms).sum() / weights.sum()
            else:
                behaviour_loss = self.loss_functions['b_loss_func'](upper_bs[0], b_train[:, 0].long())
        else:
            # b_train: (m, num_beh) at n_steps=1 or (m, n_steps, num_beh) at n_steps>1
            if b_train.dim() == 2:
                b_train = b_train.unsqueeze(1)  # → (m, 1, num_beh)
            if unroll in ('both', 'behaviour'):
                b_terms = torch.stack([
                    self.loss_functions['b_loss_func'](b_train[:, j].float(), upper_bs[j])
                    for j in range(self.n_steps)
                ])
                behaviour_loss = (weights * b_terms).sum() / weights.sum()
            else:
                behaviour_loss = self.loss_functions['b_loss_func'](b_train[:, 0].float(), upper_bs[0])

        total_loss = self.gamma * DCC_loss + (1 - self.gamma) * behaviour_loss
        return self.gamma * DCC_loss, (1 - self.gamma) * behaviour_loss, total_loss
