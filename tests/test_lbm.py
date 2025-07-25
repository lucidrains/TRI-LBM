import pytest

def test_lbm():
    import torch
    from TRI_LBM.lbm import LBM

    lbm = LBM(
      action_dim = 20
    )

    commands = ['pick up the apple']
    images = torch.randn(1, 3, 256, 256)
    actions = torch.randn(1, 16, 20)

    loss = lbm(
        text = commands,
        images = images,
        actions = actions
    )

    sampled_actions = lbm.sample(
        text = commands,
        images = images
    ) # (1, 16, 20)

    assert sampled_actions.shape == (1, 16, 20)
