import torch

from hiergeo.train import multi_positive_loss, scene_id


def test_scene_id():
    assert scene_id("/x/images_batche_1/00964b34_y1536_x1024.png") == "00964b34"
    assert scene_id("abc_def_y0_x512.jpg") == "abc_def"


def test_multi_positive_targets():
    groups = torch.tensor([0, 0, 1])
    big = 20.0
    # logits that agree with the grouping give a near-zero loss
    good = torch.tensor([[big, big, -big], [big, big, -big], [-big, -big, big]])
    bad = -good
    assert multi_positive_loss(good, groups) < 1e-6
    assert multi_positive_loss(bad, groups) > 10.0
