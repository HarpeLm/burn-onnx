#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "numpy",
# ]
# ///

# used to generate model: gathernd_bool.onnx
#
# GatherND on bool data (burn's gather_nd now handles bool directly). Expected values from
# onnx.reference.ReferenceEvaluator.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    data = helper.make_tensor_value_info("data", TensorProto.BOOL, [2, 3])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [3, 2])
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [3])
    node = helper.make_node("GatherND", ["data", "indices"], ["output"])
    graph = helper.make_graph([node], "gathernd_bool_graph", [data, indices], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "gathernd_bool.onnx")

    feed = {
        "data": np.array([[True, False, True], [False, True, False]]),
        "indices": np.array([[0, 1], [1, 1], [-1, 0]], dtype=np.int64),
    }
    (result,) = ReferenceEvaluator(model).run(None, feed)
    print("output", result.tolist())


if __name__ == "__main__":
    main()
