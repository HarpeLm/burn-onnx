#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: split_shape_sizes.onnx
#
# Split into two outputs whose sizes come from a runtime Shape: the
# dimensions of `s` are dynamic, so Shape(s) is only known at run time.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [6])
    s = helper.make_tensor_value_info("s", TensorProto.FLOAT, [None, None])
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [None])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [None])

    shape = helper.make_node("Shape", ["s"], ["sizes"])
    split = helper.make_node("Split", ["x", "sizes"], ["a", "b"], axis=0)
    graph = helper.make_graph([shape, split], "split_shape_sizes_graph", [x, s], [a, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "split_shape_sizes.onnx")

    ref = ReferenceEvaluator(model)
    data = np.arange(6, dtype=np.float32)
    for dims in ([2, 4], [5, 1]):
        s_in = np.zeros(dims, dtype=np.float32)
        a_out, b_out = ref.run(None, {"x": data, "s": s_in})
        print(f"{dims}: {a_out.tolist()} {b_out.tolist()}")


if __name__ == "__main__":
    main()
