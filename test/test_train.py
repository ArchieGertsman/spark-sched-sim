from cfg_loader import load
from trainers import make_trainer


def test_train():
    cfg = load("test/test.yaml")
    make_trainer(cfg).train()
