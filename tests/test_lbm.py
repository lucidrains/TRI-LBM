import pytest
param = pytest.mark.parametrize

@param('tactile', (False, True))
@param('diffusion_steer', (False, True))
def test_lbm(
    tactile,
    diffusion_steer
):
    import torch
    from TRI_LBM.lbm import LBM

    lbm = LBM(
        action_dim = 20,
        dim_pose = 4,
        dim_tactile_input = 37 if tactile else None
    )

    commands = ['pick up the apple']
    images = torch.randn(1, 3, 3, 224, 224)
    actions = torch.randn(1, 16, 20)
    pose = torch.randn(1, 4)

    touch = torch.randn(1, 2, 37) if tactile else None

    loss = lbm(
        text = commands,
        images = images,
        actions = actions,
        pose = pose,
        touch = touch
    )

    sampled_out = lbm.sample(
        text = commands,
        images = images,
        pose = pose,
        touch = touch,
        return_noise = diffusion_steer
    )

    if not diffusion_steer:
        sampled_actions = sampled_out
        assert sampled_actions.shape == (1, 16, 20)
    else:
        sampled_actions, noise = sampled_out
        assert sampled_actions.shape == noise.shape
