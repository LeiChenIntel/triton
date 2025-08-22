import functools
import hashlib

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

from triton._C.libtriton import xpu, ir, passes
from triton.backends.compiler import BaseBackend, GPUTarget

print("xpu backend compiler")


@dataclass(frozen=True)
class XPUOptions:
    debug: bool = False
    cluster_dims: tuple = (1, 1, 1)
    backend_name: str = "xpu"

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class XPUBackend(BaseBackend):
    # Need to define abstract methods in BaseBackend
    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "xpu"

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        self.binary_ext = "so"  # This is a fake extension to enable pipeline

    def parse_options(self, opts) -> Any:
        return XPUOptions()

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        return

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self):
        return

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        metadata["name"] = "xpu_kernel"
        return mod

    def add_stages(self, stages, options):
        # Add the processing stages for the XPU backend
        # Triton -> TritonXPU -> Lower assembly
        # Pass interfaces are implemented in the triton_xpu.cc.
        print("Adding stages for XPU backend")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        return

    @functools.lru_cache()
    def hash(self):
        return f'xpu-xx'
