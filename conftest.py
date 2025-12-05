import os

def pytest_configure(config):
    os.environ["TRITON_INTERPRET"] = "1"
