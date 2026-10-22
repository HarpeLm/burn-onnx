#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "numpy",
# ]
# ///

# used to generate model: cumprod_shape.onnx
#
# CumProd (opset 26) on the output of Shape over an input with dynamic dimensions, so the
# data is an int64 vector known only at run time: once with a constant axis 0 and once with
# a runtime axis, exclusive and reverse. Expected values from onnx.reference.ReferenceEvaluator.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, "W"])
    axis = helper.make_tensor_value_info("axis", TensorProto.INT64, [])
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.INT64, [3])
        for name in ("y", "y_exclusive_reverse")
    ]
    constant_axis = numpy_helper.from_array(np.array(0, dtype=np.int64), name="constant_axis")
    nodes = [
        helper.make_node("Shape", ["x"], ["shape"]),
        helper.make_node("CumProd", ["shape", "constant_axis"], ["y"]),
        helper.make_node("CumProd", ["shape", "axis"], ["y_exclusive_reverse"], exclusive=1, reverse=1),
    ]
    graph = helper.make_graph(nodes, "cumprod_shape_graph", [x, axis], outputs, [constant_axis])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 26)])
    model.ir_version = 11
    onnx.checker.check_model(model)
    onnx.save(model, "cumprod_shape.onnx")

    feed = {"x": np.zeros((2, 3, 4), dtype=np.float32), "axis": np.array(-1, dtype=np.int64)}
    y, y_exclusive_reverse = ReferenceEvaluator(model).run(None, feed)
    print("y", y.tolist(), "y_exclusive_reverse", y_exclusive_reverse.tolist())


if __name__ == "__main__":
    main()
