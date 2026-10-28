# PoC files

| File | Purpose |
|---|---|
| `make_poc.py` | Generates `poc.onnx` (needs the `onnx` Python package). |
| `poc.onnx` | One initializer: `data_type=22` (INT4), `dims=[2]`, `int32_data` of 2048 elements. 2115 bytes. |
| `poc_c1_arith.cpp` | Standalone harness. The size arithmetic of `getTensorOrWeightsSizeBytes`, `createTempWeights` and `convertPackedInt32Data` (release/10.16-GA) is reproduced without the TensorRT headers and driven with the values `poc.onnx` produces. |
| `asan-arith-harness.txt` | AddressSanitizer output of `poc_c1_arith.cpp`. |
| `scan_int4_initializers.py` | Standard-library scanner that flags INT4/FP4 initializers whose data length exceeds the shape-derived size. |

Build and run the harness:

```bash
g++ -std=c++20 -g -fsanitize=address -o poc_c1_arith poc_c1_arith.cpp
./poc_c1_arith
```

Scan a model:

```bash
python3 scan_int4_initializers.py poc.onnx
```

Exit status is 1 and one `SUSPICIOUS` line is printed per offending initializer. Only top-level graph initializers are inspected; subgraph initializers and `Constant` node tensors are not.

The harness does not link TensorRT. End-to-end reproduction through `nvonnxparser` is not included.
