#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with a Clip whose runtime `min` is not a scalar.

Tests:
- Clip min/max must be scalars (spec); a non-scalar runtime bound is rejected
  at import with a ProcessError instead of panicking in codegen.
"""

import onnx
from onnx import TensorProto, helper


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    min_value = helper.make_tensor_value_info("min_value", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])

    node = helper.make_node("Clip", ["x", "min_value", ""], ["y"], name="clip_tensor_min")
    graph = helper.make_graph([node], "clip_non_scalar_runtime_bound", [x, min_value], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "../fixtures/clip_non_scalar_runtime_bound.onnx")


if __name__ == "__main__":
    main()
