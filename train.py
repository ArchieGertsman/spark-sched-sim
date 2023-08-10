from cfg_loader import load
from trainers import make_trainer


if __name__ == '__main__':
    cfg = load()
    make_trainer(cfg).train()