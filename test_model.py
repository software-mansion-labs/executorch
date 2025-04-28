from pathlib import Path

import torch
from executorch.runtime import Verification, Runtime, Program, Method

et_runtime: Runtime = Runtime.get()
program: Program = et_runtime.load_program(
    Path("./xnnpack_craft_320.pte"),
    verification=Verification.Minimal,
)

forward: Method = program.load_method("forward")

inputs = (torch.ones(1, 3, 1280, 320),)
outputs = forward.execute(inputs)
