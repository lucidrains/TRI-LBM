import pytest
param = pytest.mark.parametrize

@param('tactile', (False, True))
@param('test_task_status', (False, True))
@param('diffusion_steer', (False, True))
def test_lbm(
    tactile,
    test_task_status,
    diffusion_steer
):
    import torch
    from TRI_LBM.lbm import LBM

    lbm = LBM(
        action_dim = 20,
        dim_pose = 4,
        dim_tactile_input = 37 if tactile else None,
        add_task_status_prediction = test_task_status,
    )

    commands = ['pick up the apple']
    images = torch.randn(1, 3, 3, 224, 224)
    actions = torch.randn(1, 16, 20)
    pose = torch.randn(1, 4)

    touch = torch.randn(1, 2, 37) if tactile else None

    task_status = torch.randint(-1, 2, (1,)) if test_task_status else None

    loss = lbm(
        text = commands,
        images = images,
        actions = actions,
        pose = pose,
        touch = touch,
        task_status = task_status
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
