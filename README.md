# onnx-tensorrt: heap out-of-bounds write in the INT4/FP4 initializer path

`WeightsContext::convertOnnxWeights` in onnx-tensorrt allocates the destination buffer for a packed sub-byte initializer (`INT4`, `FLOAT4E2M1`) from the tensor shape, then copies a number of bytes taken from the length of the `int32_data` field. Both values come from the model file and are never reconciled. A crafted initializer with a small `dims` and a long `int32_data` writes past the end of the heap allocation while the model is being parsed, before any builder or GPU step.

The guard was added upstream in `release/11.0-GA`. This repository documents the issue for versions up to and including `release/10.16-GA`.

| | |
|---|---|
| CVE | None assigned. See [Prior coverage](#prior-coverage). |
| CWE | CWE-787 Out-of-bounds Write (heap) |
| Project | [onnx/onnx-tensorrt](https://github.com/onnx/onnx-tensorrt) (ONNX parser used by TensorRT) |
| Affected | `release/10.0-GA` through `release/10.16-GA` (`WeightsContext.cpp`, sink present and guard absent in every branch checked, see [Scope](#scope-stated-plainly)) |
| Fixed in | `release/11.0-GA` (commit `5ab48c5c436a5686b8574c4a372cbca224273b5a`, 2026-06-02). `release/11.1-GA` to `release/11.4-GA` and `main` (`8f60063873dd`) also carry the guard. |
| CVSS v3.1 | `CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:N/I:L/A:H` = 6.1 (Medium) |
| Reported by | s1ko ([github.com/s1ko](https://github.com/s1ko), `s1ko@riseup.net`) |

## Scope, stated plainly

- Triggered by parsing a model file. The victim must load the model (`UI:R`). Network reachability depends on the deployment: a service that accepts uploaded `.onnx` files and converts them with TensorRT is exposed remotely; a developer opening a downloaded model is exposed locally.
- Demonstrated: a deterministic AddressSanitizer `heap-buffer-overflow` WRITE of size 1, 2047 bytes past a 1-byte allocation, in a standalone harness that reproduces the parser's size arithmetic verbatim. Code execution is not demonstrated. The bytes written are the low 8 bits of attacker-chosen `int32` values and the length is attacker-chosen, so the primitive is not limited to a crash, but exploitability depends on heap layout and was not assessed.
- Not demonstrated: an end-to-end run of `poc.onnx` through a TensorRT-linked build of the parser. UNVERIFIED. The call chain was established by reading the source (below).
- Version range: the sink was read in `WeightsContext.cpp` of `release/10.0-GA` to `release/10.16-GA` (every branch in between was fetched; none contains the guard). Earlier branches were not checked for the packed INT4 path; `release/9.3` does not contain `convertPackedInt32Data`.
- Not a fix credit: the upstream change that adds the guard is the squashed commit "ONNX-TRT 11.0 GA Release (#1054)". It does not reference this report or any advisory.

## Threat model

Attacker supplies an `.onnx` file (model hub download, upload to a conversion service, CI artifact). The victim parses it with `nvonnxparser` / `onnx2trt` / `trtexec --onnx`. No weights file, GPU, builder, or inference is needed. The malformed field is a normal ONNX `TensorProto`; it is well-formed protobuf and survives a round trip through the `onnx` Python package.

## Execution

### Root cause

All excerpts are from `release/10.16-GA` (`6bde6596eca84abf9afe9b4d21c18c32800fd3ad`, 2026-03-25), file `WeightsContext.cpp`.

Call site, lines 395-401 (`convertOnnxWeights`, in the branch that handles initializers stored in `int32_data`):

```cpp
            case ::ONNX_NAMESPACE::TensorProto::INT4:
                // int4 data is packed, each int32 element contains one byte (two int4 nibbles)
            case ::ONNX_NAMESPACE::TensorProto::FLOAT4E2M1:
                // int4/fp4 data is packed, each int32 element contains one byte (two int4/fp4 nibbles)
                nbytes = onnxTensor.int32_data().size();
                dataPtr = convertPackedInt32Data(onnxTensor.int32_data().data(), shape, nbytes, onnxDtype);
                break;
```

Sink, lines 73-83:

```cpp
uint8_t* WeightsContext::convertPackedInt32Data(
    int32_t const* weightValues, nvinfer1::Dims const& shape, size_t nbytes, int32_t onnxdtype)
{
    uint8_t* newWeights{static_cast<uint8_t*>(createTempWeights(onnxdtype, shape).values)};

    for (size_t i = 0; i < nbytes; i++)
    {
        newWeights[i] = static_cast<uint8_t>(weightValues[i]);
    }
    return newWeights;
}
```

Allocation, lines 509-521:

```cpp
ShapedWeights WeightsContext::createTempWeights(ShapedWeights::DataType type, nvinfer1::Dims const& shape)
{
    ShapedWeights weights(type, nullptr, shape);
    int64_t nbBytes = weights.size_bytes();
    // For empty weights, keep the values as nullptr.
    if (nbBytes == 0)
    {
        return weights;
    }
    // Create a zero'd buffer (`std::make_unique<T[]>(n)` uses value initialization):
    weights.values = mWeightBuffers.emplace_back(std::make_unique<std::byte[]>(nbBytes)).get();
    return weights;
}
```

`size_bytes()` resolves to `getTensorOrWeightsSizeBytes(count, type)` in `weightUtils.cpp`, which for a 4-bit type is `ceil(count * 4 / 8)`. For `dims=[2]` that is 1 byte. The loop bound `nbytes` is the raw element count of the repeated `int32_data` field (2048 in the PoC) and is not derived from the shape.

Why the existing checks do not stop it, all in the same file at the same tag:

- `validateOnnxInitializer` checks dtype, rank and `volume * typeSize` overflow. It never looks at the length of any data field.
- `getInitializerVol` and `ShapedWeights::count()` only guard overflow of the dims product.
- The size comparison for external weights runs only on the external-data branch.

A second trigger by inspection (not executed): `dims=[0]` yields `nbBytes == 0`, `createTempWeights` returns `values == nullptr`, and any non-empty `int32_data` makes the loop write through a null pointer.

### Fix upstream

`release/11.0-GA`, `WeightsContext.cpp:80-97`:

```cpp
uint8_t* WeightsContext::convertPackedInt32Data(
    int32_t const* weightValues, nvinfer1::Dims const& shape, size_t nbytes, int32_t onnxdtype)
{
    auto* ctx = this; // For logging macros.
    ShapedWeights newWeightsObj{createTempWeights(onnxdtype, shape)};
    if (nbytes > newWeightsObj.size_bytes())
    {
        LOG_ERROR("Packed int32 data size (" << nbytes << " bytes) exceeds the shape-derived allocation size ("
            << newWeightsObj.size_bytes() << " bytes). Rejecting malformed weights.");
        return nullptr;
    }
    ...
```

The same release adds an equivalent check to `ownWeights` (the `raw_data` memcpy path, `WeightsContext.cpp:15-21` in 10.16-GA), which had the same shape-versus-data divergence. The read-side siblings are guarded later: in `release/11.0-GA` the `convertInt32Data` template in `WeightsContext.hpp` still loops over `volume(shape)` without comparing it to the source length, and by `release/11.4-GA` both `convertInt32Data` and `convertDouble` reject a source shorter than `volume(shape)`.

### Reproduction

Model generation (`poc/make_poc.py`, or use the shipped `poc/poc.onnx`):

```bash
python3 poc/make_poc.py
```

Resulting `TensorProto`: `data_type = 22` (INT4), `dims = [2]`, `int32_data` of length 2048 (two int4 nibbles per element, 4096 logical values).

Standalone harness with AddressSanitizer:

```bash
cd poc
g++ -std=c++17 -g -fsanitize=address -o poc_c1_arith poc_c1_arith.cpp
./poc_c1_arith
```

Observed (`poc/asan-arith-harness.txt`):

```
[poc] createTempWeights: dtype=22 shapeCount=2 -> alloc 1 byte(s)
[poc] convertPackedInt32Data: writing nbytes=2048 into the buffer above
==ERROR: AddressSanitizer: heap-buffer-overflow ... WRITE of size 1
    #0 ... in convertPackedInt32Data ... poc_c1_arith.cpp:69
0x502000000011 is located 0 bytes after 1-byte region [0x502000000010,0x502000000011)
```

Against a real build (not executed here): build onnx-tensorrt `release/10.16-GA` against the matching TensorRT, then `onnx2trt poc/poc.onnx -o /dev/null`, preferably with `-fsanitize=address` on the parser library. Expected: the parse aborts or corrupts the heap in `convertPackedInt32Data` before the builder starts.

## Detection

File-format check (no TensorRT needed): for every `TensorProto` with `data_type` 22 (INT4) or 23 (FLOAT4E2M1), compute `expected = ceil(prod(dims) * 4 / 8)` and flag the file when `len(int32_data) > expected` or `len(raw_data) > expected`. `poc/scan_int4_initializers.py` implements this for top-level graph initializers:

```
$ python3 poc/scan_int4_initializers.py poc/poc.onnx
SUSPICIOUS w dtype=22 dims=[2] expected_bytes=1 int32_data=2048 raw_data=0
```

Equivalent with the `onnx` package, including subgraphs and `Constant` attributes, is a walk over `onnx.helper` tensors comparing `len(t.int32_data)` against `(prod(t.dims) * 4 + 7) // 8` for `t.data_type in (onnx.TensorProto.INT4, onnx.TensorProto.FLOAT4E2M1)`.

Runtime signatures:

- AddressSanitizer: `heap-buffer-overflow ... WRITE of size 1` with top frame `onnx2trt::WeightsContext::convertPackedInt32Data` (`WeightsContext.cpp`) and the allocation frame `onnx2trt::WeightsContext::createTempWeights`.
- On `release/11.0-GA` and later the parser logs `Packed int32 data size (N bytes) exceeds the shape-derived allocation size (M bytes). Rejecting malformed weights.` and the import fails. That log line in a conversion pipeline marks an attempt to feed this file shape.

## Mitigation

- Upgrade the ONNX parser to `release/11.0-GA` or later (or the TensorRT package that bundles it). The mapping between TensorRT package versions and parser branches was not verified here. UNVERIFIED.
- If upgrading is not possible, reject models before parsing: run a check such as `poc/scan_int4_initializers.py` and refuse files with INT4/FP4 initializers whose data is longer than the shape implies; or backport the size check from `release/11.0-GA` to `convertPackedInt32Data` and `ownWeights`.
- Convert untrusted models in a sandboxed, memory-limited process without access to credentials.
- Mapping: CWE-787 (validate destination size before copying); ATT&CK M1048 Application Isolation and Sandboxing and D3FEND Process Isolation for the conversion service.

## Timeline

| Date (UTC) | Event |
|---|---|
| 2026-03-25 | `release/10.16-GA` head `6bde6596`, used for the original reproduction. |
| 2026-05-20 | CVE-2026-24188 published (NVIDIA TensorRT, out-of-bounds write; component not stated in the record). |
| 2026-05-31 | Reported via huntr.com (report `06bd1459-1b2c-4180-94f1-3f819d17c70f`). No triage, validation or response from the platform as of 2026-10-28. |
| 2026-06-02 | Upstream commit `5ab48c5c` "ONNX-TRT 11.0 GA Release (#1054)" adds the size guard to `convertPackedInt32Data` and `ownWeights`. |
| 2026-10-28 | Re-checked: `release/11.0-GA` to `release/11.4-GA` and `main` carry the guard; `release/10.16-GA` and earlier do not. This advisory published. |

## Prior coverage

- Upstream fixed the code path in `release/11.0-GA` (see above). No release note, issue, or advisory linking the change to a report was found.
- CVE-2026-24188 ([GHSA-pffw-vmg5-5gcr](https://github.com/advisories/GHSA-pffw-vmg5-5gcr), published 2026-05-20, CWE-787, CVSS 8.2 assigned by NVIDIA) is "NVIDIA TensorRT contains a vulnerability where an attacker could cause an out-of-bounds write". The record names no component or function, and the NVIDIA bulletin referenced by it (`a_id/5836`) could not be retrieved when this was written. The original report treated this issue as the same bug class as that CVE. This advisory does not claim that CVE covers `convertPackedInt32Data`, and does not claim it does not. No CVE is claimed for this finding.
- No third-party public write-up of this sink was found.

## References

- Affected code: [`WeightsContext.cpp` @ release/10.16-GA](https://github.com/onnx/onnx-tensorrt/blob/6bde6596eca84abf9afe9b4d21c18c32800fd3ad/WeightsContext.cpp#L73-L83)
- Fixed code: [`WeightsContext.cpp` @ release/11.0-GA](https://github.com/onnx/onnx-tensorrt/blob/5ab48c5c436a5686b8574c4a372cbca224273b5a/WeightsContext.cpp#L80-L97)
- [CVE-2026-24188](https://nvd.nist.gov/vuln/detail/CVE-2026-24188)
- [CWE-787](https://cwe.mitre.org/cgi-bin/jumpmenu.cgi?id=787)
- [ONNX `TensorProto` definition](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto)

## License

MIT, see [LICENSE](LICENSE).
