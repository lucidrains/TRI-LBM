import pytest
param = pytest.mark.parametrize

@param('tactile', (False, True))
@param('test_task_status', (False, True))
@param('diffusion_steer', (False, True))
@param('additional_context', (False, True))
@param('send_vlm_key_values', (False, True))
@param('use_depth_embed', (False, True))
@param('cross_attend_text_encodings', (False, True))
def test_lbm(
    tactile,
    test_task_status,
    diffusion_steer,
    additional_context,
    send_vlm_key_values,
    use_depth_embed,
    cross_attend_text_encodings
):
    import torch
    from TRI_LBM.lbm import LBM

    lbm = LBM(
        action_dim = 20,
        dim_pose = 4,
        dim_tactile_input = 37 if tactile else None,
        add_task_status_prediction = test_task_status,
        accept_additional_context = additional_context,
        depth = 1,
        dim = 64,
        additional_context_dim = 17,
        dim_depth_embed = 21 if use_depth_embed else None,
        cross_attend_text_encodings = cross_attend_text_encodings
    )

    commands = ['pick up the apple', 'put down the book']
    images = torch.randn(2, 3, 3, 224, 224)
    actions = torch.randn(2, 16, 20)
    pose = torch.randn(2, 4)

    context = torch.randn(2, 32, 17) if additional_context else None
    context_mask = torch.randint(0, 2, (2, 32)).bool() if additional_context else None

    touch = torch.randn(2, 2, 37) if tactile else None

    task_status = torch.randint(-1, 2, (2,)) if test_task_status else None

    vlm_key_values = None
    if send_vlm_key_values:
        vlm_key_values = [
            (torch.randn(2, 12, 32, 64), torch.randn(2, 12, 32, 64)),
            (torch.randn(2, 12, 32, 64), torch.randn(2, 12, 32, 64)),
        ]

    depth_embed = None
    if use_depth_embed:
        depth_embed = torch.randn(2, 21)

    loss = lbm(
        text = commands,
        images = images,
        actions = actions,
        pose = pose,
        touch = touch,
        context = context,
        context_mask = context_mask,
        task_status = task_status,
        vlm_key_values = vlm_key_values,
        depth_embed = depth_embed
    )

    sampled_out = lbm.sample(
        text = commands,
        images = images,
        pose = pose,
        touch = touch,
        context = context,
        context_mask = context_mask,
        return_noise = diffusion_steer,
        vlm_key_values = vlm_key_values,
        depth_embed = depth_embed
    )

    if not diffusion_steer:
        sampled_actions = sampled_out
        assert sampled_actions.shape == (2, 16, 20)
    else:
        sampled_actions, noise = sampled_out
        assert sampled_actions.shape == noise.shape

@param('parallel', (False, True))
def test_welford(parallel):
    import torch
    from TRI_LBM.lbm import ActionClassifier

    classifier = ActionClassifier(
        dim_action = 20,
        num_action_types = 1
    )

    actions = torch.randn(128, 20)
    action_types = torch.randint(0, 1, (128,))

    if parallel:
        for actions_chunk, action_types_chunk in zip(actions.chunk(2, dim = 0), action_types.chunk(2, dim = 0)):
            classifier.update_action_statistics_with_parallel_welford_(actions_chunk, action_types_chunk)

    else:
        classifier.update_action_statistics_with_welford_(actions, action_types)

    assert torch.allclose(classifier.action_mean, actions.mean(dim = 0))

    assert torch.allclose(classifier.action_variance, actions.var(dim = 0))

def test_action_norm():
    import torch
    from TRI_LBM.lbm import ActionClassifier

    classifier = ActionClassifier(
        dim_action = 20,
        num_action_types = 7
    )

    action_chunks = torch.randn(2, 12, 20)
    action_types = torch.randint(0, 7, (2,))

    classifier.update_action_statistics_with_parallel_welford_(action_chunks, action_types)

    loss = classifier(action_chunks, action_types)
    loss.backward()

    normed_action_chunks = classifier.normalize(action_chunks, action_types)

    pred_classes, unnormed_actions_to_robot = classifier(normed_action_chunks)

def test_lbm_with_action_classifier():
    import torch
    from TRI_LBM.lbm import LBM, ActionClassifier

    action_classifier = ActionClassifier(
        dim_action = 21,  # 20 + 1 for task status
        num_action_types = 3
    )

    dummy_actions = torch.randn(128, 16, 21)
    dummy_action_types = torch.randint(0, 3, (128,))
    action_classifier.update_action_statistics_with_parallel_welford_(dummy_actions, dummy_action_types)

    lbm = LBM(
        action_dim = 20,
        dim_pose = 4,
        action_chunk_normalizer = action_classifier,
        depth = 1,
        dim = 64,
        add_task_status_prediction = True
    )

    commands = ['pick up the apple', 'put down the book']
    images = torch.randn(2, 3, 3, 224, 224)
    actions = torch.randn(2, 16, 20)
    pose = torch.randn(2, 4)

    action_types = torch.randint(0, 3, (2,))
    task_status = torch.randint(-1, 2, (2,))

    loss = lbm(
        text = commands,
        images = images,
        actions = actions,
        pose = pose,
        action_types = action_types,
        task_status = task_status
    )

    sampled_actions = lbm.sample(
        text = commands,
        images = images,
        pose = pose
    )

    assert sampled_actions.shape == (2, 16, 20)
