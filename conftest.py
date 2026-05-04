import os, torch

def pytest_configure(config):
    if not torch.cuda.is_available():
        os.environ["TRITON_INTERPRET"] = "1"
