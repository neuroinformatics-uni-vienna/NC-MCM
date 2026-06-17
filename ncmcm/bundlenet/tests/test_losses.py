import torch
import functools
from ncmcm.bundlenet.losses import BccDccLoss


def test_BccDccLoss_discrete():
    yt1_upper = torch.randn(3, 5)
    yt1_lower = torch.randn(3, 5)
    bt1_upper = torch.randn(3, 5)
    b_train_1 = torch.empty(3, dtype=torch.long).random_(5)
    gamma = torch.rand(1)
    b_type = 'discrete'

    bccdcc_loss = BccDccLoss(b_type, gamma)
    dcc_loss, behaviour_loss, total_loss = bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_train_1)

    # MSELoss * gamma
    expected_dcc_loss = torch.mean((yt1_upper - yt1_lower) ** 2) * gamma

    # CrossEntropyLoss * (1 - gamma)
    log_softmax = torch.log_softmax(bt1_upper, dim=-1)
    expected_behaviour_loss = -log_softmax[range(len(log_softmax)), b_train_1].mean() * (1 - gamma)

    torch.testing.assert_close(dcc_loss, expected_dcc_loss)
    torch.testing.assert_close(behaviour_loss, expected_behaviour_loss)
    torch.testing.assert_close(total_loss, expected_dcc_loss + expected_behaviour_loss)


def test_BccDccLoss_continuous():
    yt1_upper = torch.randn(3, 5)
    yt1_lower = torch.randn(3, 5)
    bt1_upper = torch.randn(3, 5)
    b_train_1 = torch.randn(3, 5)
    gamma = torch.rand(1)
    b_type = 'continuous'

    bccdcc_loss = BccDccLoss(b_type, gamma)
    dcc_loss, behaviour_loss, total_loss = bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_train_1)

    # MSELoss * gamma
    expected_dcc_loss = torch.mean((yt1_upper - yt1_lower) ** 2) * gamma

    # MSELoss * (1 - gamma)
    expected_behaviour_loss = torch.mean((bt1_upper - b_train_1) ** 2) * (1 - gamma)

    torch.testing.assert_close(dcc_loss, expected_dcc_loss)
    torch.testing.assert_close(behaviour_loss, expected_behaviour_loss)
    torch.testing.assert_close(total_loss, expected_dcc_loss + expected_behaviour_loss)


def test_BccDccLoss_gamma_extremes():
    yt1_upper = torch.randn(3, 5)
    yt1_lower = torch.randn(3, 5)
    bt1_upper = torch.randn(3, 5)
    b_train_1 = {
        'discrete': torch.empty(3, dtype=torch.long).random_(5),
        'continuous': torch.randn(3, 5) 
    }

    for gamma in [0.0, 1.0]:
        for b_type in ['discrete', 'continuous']:
            bccdcc_loss = BccDccLoss(b_type, gamma=gamma)
            dcc, bcc, total = bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_train_1[b_type])

            if gamma == 0.0:
                assert torch.isclose(total, bcc)
                assert torch.isclose(dcc, torch.tensor(0.0))
            elif gamma == 1.0:
                assert torch.isclose(total, dcc)
                assert torch.isclose(bcc, torch.tensor(0.0))
            assert torch.isclose(total, gamma * dcc + (1 - gamma) * bcc)