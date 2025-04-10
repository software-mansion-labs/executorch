# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import binascii
import os
from argparse import ArgumentParser, ArgumentTypeError

# Also see: https://git.gitlab.arm.com/artificial-intelligence/ethos-u/ml-embedded-evaluation-kit/-/blob/main/scripts/py/gen_model_cpp.py?ref_type=heads

bytes_per_line = 32
hex_digits_per_line = bytes_per_line * 2


def input_file_path(path):
    if os.path.exists(path):
        return path
    else:
        raise ArgumentTypeError(f"input filepath:{path} does not exist")


parser = ArgumentParser()
parser.add_argument(
    "-p",
    "--pte",
    help="ExecuTorch .pte model file",
    type=input_file_path,
    required=True,
)
parser.add_argument(
    "-d",
    "--outdir",
    help="Output dir for model header",
    type=str,
    required=False,
    default=".",
)

parser.add_argument(
    "-o",
    "--outfile",
    help="Output filename for model header",
    type=str,
    required=False,
    default="model_pte.h",
)
parser.add_argument(
    "-s",
    "--section",
    help="Section attribute for the data array",
    type=str,
    required=False,
    default="network_model_sec",
)

if __name__ == "__main__":
    args = parser.parse_args()
    outfile = os.path.join(args.outdir, args.outfile)
    attr = f'__attribute__((section("{args.section}"), aligned(16))) char '

    with open(args.pte, "rb") as fr, open(outfile, "w") as fw:
        data = fr.read()
        hexstream = binascii.hexlify(data).decode("utf-8")
        fw.write(attr + "model_pte[] = {")

        for i in range(0, len(hexstream), 2):
            if 0 == (i % hex_digits_per_line):
                fw.write("\n")
            fw.write("0x" + hexstream[i : i + 2] + ", ")

        fw.write("};\n")

        print(
            f"Input: {args.pte} with {len(data)} bytes. Output: {outfile} with {fw.tell()} bytes. Section: {args.section}."
        )
