from triton.backends.compiler import BaseBackend


class XPUBackend(BaseBackend):
    def __init__(self):
        super().__init__()
        self.name = "xpu"
