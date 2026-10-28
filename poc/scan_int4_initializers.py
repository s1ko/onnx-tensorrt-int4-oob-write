import sys


def varint(buf, i):
    shift = 0
    value = 0
    while True:
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, i
        shift += 7


def fields(buf):
    i = 0
    while i < len(buf):
        key, i = varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = varint(buf, i)
        elif wire == 2:
            size, i = varint(buf, i)
            value = buf[i:i + size]
            i += size
        elif wire == 1:
            value = buf[i:i + 8]
            i += 8
        elif wire == 5:
            value = buf[i:i + 4]
            i += 4
        else:
            raise ValueError("unsupported wire type %d" % wire)
        yield number, wire, value


def packed_varints(buf):
    i = 0
    count = 0
    while i < len(buf):
        _, i = varint(buf, i)
        count += 1
    return count


def inspect_tensor(buf):
    dims = []
    dtype = 0
    name = b""
    int32_count = 0
    raw_len = 0
    for number, wire, value in fields(buf):
        if number == 1:
            if wire == 0:
                dims.append(value)
            else:
                j = 0
                while j < len(value):
                    d, j = varint(value, j)
                    dims.append(d)
        elif number == 2:
            dtype = value
        elif number == 5:
            int32_count += packed_varints(value) if wire == 2 else 1
        elif number == 8:
            name = value
        elif number == 9:
            raw_len = len(value)
    return name.decode("utf8", "replace"), dtype, dims, int32_count, raw_len


def main(path):
    model = open(path, "rb").read()
    bad = 0
    for number, _, graph in fields(model):
        if number != 7:
            continue
        for gnum, _, tensor in fields(graph):
            if gnum != 5:
                continue
            name, dtype, dims, int32_count, raw_len = inspect_tensor(tensor)
            if dtype not in (22, 23):
                continue
            volume = 1
            for d in dims:
                volume *= d
            expected = (volume * 4 + 7) // 8
            if int32_count > expected or raw_len > expected:
                bad += 1
                print(
                    "SUSPICIOUS %s dtype=%d dims=%s expected_bytes=%d int32_data=%d raw_data=%d"
                    % (name, dtype, dims, expected, int32_count, raw_len)
                )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
