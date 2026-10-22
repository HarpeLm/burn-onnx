#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "numpy",
# ]
# ///

# used to generate model: cumprod_int32.onnx
#
# CumProd (opset 26) on an int32 [2, 3] input with a constant axis 1, so the axis is known at
# import time. Exclusive fills with ones and the output stays int32. Expected values from
# onnx.reference.ReferenceEvaluator.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

VARIANTS = [
    ("y_exclusive", {"exclusive": 1}),
    ("y_exclusive_reverse", {"exclusive": 1, "reverse": 1}),
]


def main():
    x = helper.make_tensor_value_info("x", TensorProto.INT32, [2, 3])
    outputs = [helper.make_tensor_value_info(name, TensorProto.INT32, [2, 3]) for name, _ in VARIANTS]
    axis = numpy_helper.from_array(np.array(1, dtype=np.int64), name="axis")
    nodes = [helper.make_node("CumProd", ["x", "axis"], [name], **attrs) for name, attrs in VARIANTS]
    graph = helper.make_graph(nodes, "cumprod_int32_graph", [x], outputs, [axis])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 26)])
    model.ir_version = 11
    onnx.checker.check_model(model)
    onnx.save(model, "cumprod_int32.onnx")

    data = np.array([[2, 3, 4], [-1, 5, 2]], dtype=np.int32)
    results = ReferenceEvaluator(model).run(None, {"x": data})
    for (name, _), result in zip(VARIANTS, results):
        print(name, result.dtype, result.tolist())


if __name__ == "__main__":
    main()
