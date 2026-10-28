import onnx
from onnx import helper, TensorProto

t = helper.make_tensor("w", TensorProto.INT4, dims=[2], vals=[0] * 4096)
g = helper.make_graph(
    [helper.make_node("Identity", ["w"], ["y"])],
    "poc_c1",
    [],
    [helper.make_tensor_value_info("y", TensorProto.INT4, [2])],
    initializer=[t],
)
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)]), "poc.onnx")
