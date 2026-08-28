"""Differential harness and seed corpus generation. The replay harness, the libFuzzer target,
and the host-side decoder share one byte consumption layout, so a seed denotes the same func0
call in each. All inputs are signature-derived. The typed_decode, rich_observables, and
address_free features toggle independently."""

from __future__ import annotations

import random
import struct

from decomp.signature import (CAP, DISPLAY_ELEMS, INPUT_BUFFER, OOC_BOUND, SCALAR_FLOATS,
                              SCALAR_INTS, SCALAR_INTS_64, STR_BUF, Signature, norm_ctype)

# Shims for Ghidra pseudo-types so forward declarations using them still link.
_TYPEDEFS = """typedef unsigned char      undefined;
typedef unsigned char      undefined1;
typedef unsigned short     undefined2;
typedef unsigned int       undefined4;
typedef unsigned long long undefined8;
typedef unsigned char      byte;
typedef unsigned char      uchar;
typedef unsigned short     ushort;
typedef unsigned int       uint;
typedef unsigned long      ulong;
"""

# The byte-stream decoder shared verbatim by every harness.
_TAKE = """static unsigned char _buf[%(n)d];
static size_t _len, _cur;
static int _need(size_t n) { return (_cur + n) <= _len; }
static int32_t _take_i32(int32_t f) { if (!_need(4)) return f; int32_t v; memcpy(&v,_buf+_cur,4); _cur+=4; return v; }
static int64_t _take_i64(int64_t f) { if (!_need(8)) return f; int64_t v; memcpy(&v,_buf+_cur,8); _cur+=8; return v; }
static double  _take_double(double f) { if (!_need(8)) return f; double v; memcpy(&v,_buf+_cur,8); _cur+=8; return v; }
static float   _take_float(float f) { if (!_need(4)) return f; float v; memcpy(&v,_buf+_cur,4); _cur+=4; return v; }
static unsigned char _take_byte(unsigned char f) { if (!_need(1)) return f; return _buf[_cur++]; }
static void _take_raw(void *dst, size_t n) { unsigned char *d=dst; for (size_t i=0;i<n;++i) d[i]=(_cur<_len)?_buf[_cur++]:0; }
""" % {"n": INPUT_BUFFER}

_HEADER = """/* Auto-generated differential harness. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/prctl.h>

""" + _TYPEDEFS + """
static void _dump_ptr16(const void *p) {
    const unsigned char *b = p; printf("ptr:");
    for (int i = 0; i < 16; ++i) printf("%02x", b[i]); printf("\\n");
}

__FORWARD__
"""


def _header(forward: str) -> str:
    return _HEADER.replace("__FORWARD__", forward)


def _scalar_take(ctype: str) -> str:
    if ctype in SCALAR_FLOATS:
        return "_take_float(0.0f)" if ctype == "float" else "_take_double(0.0)"
    if ctype in ("char", "int8_t", "uint8_t"):
        return "(int)_take_byte(0)" if ctype == "int8_t" else "_take_byte(0)"
    if ctype in SCALAR_INTS_64:
        return "_take_i64(0)"
    return "_take_i32(0)"


def _print_fmt(ctype: str):
    if ctype in SCALAR_FLOATS:
        return "%.6g", "(double)"
    return "%lld", "(long long)"


def build_arg_setup(sig: Signature):
    """Per-argument C setup, call args, and output buffers for typed decode. A pointer or array
    argument becomes a CAP-element buffer that is NULL on some draws. A scalar is passed raw with
    its magnitude bounded."""
    has_buffer = any(a.is_array or a.is_string for a in sig.args)
    setup, call_args, buffers = [], [], []
    for arg in sig.args:
        ctype = arg.c_type.replace("const ", "").strip()
        if arg.is_array and not arg.is_string:
            base = ctype.rstrip("*").strip()
            if base not in SCALAR_FLOATS and base not in SCALAR_INTS:
                base = "int"
            buf = f"{arg.name}_buf"
            setup.append(f"    {base} {buf}[{CAP}] = {{0}};")
            setup.append(f"    for (int i=0;i<{CAP};++i) {buf}[i] = {_scalar_take(base)};")
            setup.append(f"    {base} *{arg.name} = (_take_byte(0) < 64) ? ({base}*)0 : {buf};")
            call_args.append(arg.name)
            buffers.append((arg.name, buf, base))
        elif arg.is_string:
            buf = f"{arg.name}_buf"
            setup.append(f"    char {buf}[{STR_BUF}] = {{0}};")
            setup.append(f"    for (int i=0;i+1<{STR_BUF};++i) {{ unsigned char c=_take_byte(0); if(!c)break; {buf}[i]=(char)c; }}")
            setup.append(f"    char *{arg.name} = (_take_byte(0) < 64) ? (char*)0 : {buf};")
            call_args.append(arg.name)
            buffers.append((arg.name, buf, "char"))
        else:
            setup.append(f"    {ctype} {arg.name} = ({ctype}){_scalar_take(ctype)};")
            if has_buffer and ctype in SCALAR_INTS:
                setup.append(f"    {arg.name} %= {OOC_BOUND + 1};")
            call_args.append(arg.name)
    return setup, call_args, buffers


def build_arg_setup_raw(sig: Signature):
    """Slice raw bytes into each argument by size. No buffers, pointers get byte-derived
    addresses."""
    setup, call_args = [], []
    for arg in sig.args:
        ctype = arg.c_type.replace("const ", "").strip()
        setup.append(f"    {ctype} {arg.name}; _take_raw(&{arg.name}, sizeof({arg.name}));")
        call_args.append(arg.name)
    return setup, call_args, []


def _ret_decl(sig: Signature):
    """Return the declared return type and whether to capture the return register. When Ghidra
    mislabels a value-returning function void and there is no buffer to observe, the harness
    captures the raw ABI return register instead."""
    capture = sig.is_void_return and not any(a.is_array or a.is_string for a in sig.args)
    return ("unsigned long long" if capture else sig.return_type), capture


def _result_print(return_type: str, address_free: bool) -> str:
    t = return_type.replace(" ", "")
    if t.endswith("*") and not address_free:
        return '    printf("ptr=%p\\n", (void *)_r);'
    if t in ("char*", "constchar*"):
        return '    printf("%s\\n", _r ? (const char *)_r : "(null)");'
    if t.endswith("*"):
        return '    if (_r) _dump_ptr16(_r); else printf("(null)\\n");'
    fmt, cast = _print_fmt(return_type)
    return f'    printf("{fmt}\\n", {cast}_r);'


def render_harness(sig, *, typed_decode=True, rich_observables=True, address_free=True) -> str:
    """The standalone replay harness. Reads stdin, decodes args, calls func0, prints observables."""
    if sig is None or not sig.args:
        return _render_no_arg(sig, address_free)

    declared, capture = _ret_decl(sig)
    forward = f"extern {declared} func0({', '.join(a.c_type for a in sig.args)});"
    setup, call_args, buffers = (build_arg_setup(sig) if typed_decode
                                 else build_arg_setup_raw(sig))
    call = ", ".join(call_args)

    if capture:
        call_line = f"    unsigned long long _r = func0({call});"
        result = ('    if (_r > 0x10000ULL) _dump_ptr16((const void*)(uintptr_t)_r); else printf("retreg=0x%llx\\n", _r);'
                  if address_free else '    printf("retreg=0x%llx\\n", _r);')
    elif sig.is_void_return:
        call_line, result = f"    func0({call});", ""
    else:
        call_line = f"    {sig.return_type} _r = func0({call});"
        result = _result_print(sig.return_type, address_free)

    dumps = []
    for _name, buf, base in (buffers if rich_observables else []):
        if base == "char":
            dumps.append(f'    printf("%.{STR_BUF}s\\n", {buf});')
        else:
            fmt, cast = _print_fmt(base)
            dumps.append(f'    for (int i=0;i<{CAP};++i) printf("{fmt} ", {cast}{buf}[i]);')
            dumps.append(r'    printf("\n");')

    body = "\n".join([call_line, result, *dumps])
    return _header(forward) + _TAKE + f"""
int main(void) {{
    prctl(PR_SET_DUMPABLE, 0);
    _len = fread(_buf, 1, sizeof(_buf), stdin); _cur = 0;
{chr(10).join(setup)}
{body}
    return 0;
}}
"""


def _render_no_arg(sig, address_free: bool) -> str:
    is_void = (sig is None) or sig.is_void_return
    if is_void:
        forward = "extern unsigned long long func0(void);"
        call = "unsigned long long _r = func0();"
        result = ('    if (_r > 0x10000ULL) _dump_ptr16((const void*)(uintptr_t)_r); else printf("retreg=0x%llx\\n", _r);'
                  if address_free else '    printf("retreg=0x%llx\\n", _r);')
    else:
        rt = sig.return_type
        forward = f"extern {rt} func0(void);"
        call = f"{rt} _r = func0();"
        result = _result_print(rt, address_free)
    return _header(forward) + (
        "int main(void) {\n    prctl(PR_SET_DUMPABLE, 0);\n"
        f"    {call}\n{result}\n    return 0;\n}}\n")


# structure-aware custom mutator, marries the typed seeds with libFuzzer's search

def _scalar_wk(ctype: str):
    """Byte width and kind of a decoded scalar, matching _scalar_take. Kind is 0 for int, 1 for
    float, 2 for double."""
    if ctype == "float":
        return 4, 1
    if ctype in ("double", "long double"):
        return 8, 2
    if ctype in ("char", "int8_t", "uint8_t"):
        return 1, 0
    if ctype in SCALAR_INTS_64:
        return 8, 0
    return 4, 0


def _slot_table(sig: Signature):
    """Offset, width, and kind of every scalar slot in the typed layout, in build_arg_setup's
    consumption order. Returns the slots and the layout size, or None and 0 when a string arg
    makes the layout variable."""
    slots, cur = [], 0
    for arg in sig.args:
        ctype = arg.c_type.replace("const ", "").strip()
        if arg.is_string:
            return None, 0                          # variable-length consumption
        if arg.is_array and not arg.is_string:
            base = ctype.rstrip("*").strip()
            if base not in SCALAR_FLOATS and base not in SCALAR_INTS:
                base = "int"
            w, k = _scalar_wk(base)
            for _ in range(CAP):
                slots.append((cur, w, k)); cur += w
            cur += 1                                 # the NULL-pointer decision byte
        else:
            w, k = _scalar_wk(ctype)
            slots.append((cur, w, k)); cur += w
    return slots, cur


def _float_lit(f: float, suffix: str) -> str:
    import math
    if math.isinf(f):
        return ("-" if f < 0 else "") + "__builtin_inf" + suffix + "()"
    if math.isnan(f):
        return "__builtin_nan" + suffix + '("")'
    return repr(f) + suffix


def _mutator_c(sig: Signature, constants) -> str:
    """A libFuzzer custom mutator that writes a boundary or original constant into a random typed
    slot, keeping mutations structurally valid. Empty when the layout is variable."""
    slots, layout = _slot_table(sig)
    if not slots:
        return ""
    ints, seen = [], set()
    for v in list(_INT_BOUNDARIES) + list(_INT64_BOUNDARIES) + list(constants or []):
        if -(1 << 63) <= v < (1 << 63) and v not in seen:
            seen.add(v); ints.append(v)
    ints = ints[:96]
    ivi = ", ".join(f"{v}LL" for v in ints)
    ivf = ", ".join(_float_lit(f, "f") for f in _FLOAT_BOUNDARIES)
    ivd = ", ".join(_float_lit(f, "") for f in _FLOAT_BOUNDARIES)
    off = ", ".join(str(o) for o, _w, _k in slots)
    wid = ", ".join(str(w) for _o, w, _k in slots)
    knd = ", ".join(str(k) for _o, _w, k in slots)
    return f"""
extern size_t LLVMFuzzerMutate(uint8_t *, size_t, size_t);
static const long long _IVI[] = {{ {ivi} }};
static const float  _IVF[] = {{ {ivf} }};
static const double _IVD[] = {{ {ivd} }};
static const int _SOFF[] = {{ {off} }};
static const int _SW[]   = {{ {wid} }};
static const int _SK[]   = {{ {knd} }};
enum {{ _NSLOT = {len(slots)}, _NIVI = {len(ints)}, _NIVF = {len(_FLOAT_BOUNDARIES)},
       _NIVD = {len(_FLOAT_BOUNDARIES)}, _LAYOUT = {layout} }};

size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size, size_t MaxSize, unsigned int Seed) {{
    if ((Seed & 3) && MaxSize >= (size_t)_LAYOUT) {{        /* 3 of 4: place a value in a typed slot */
        if (Size < (size_t)_LAYOUT) {{ for (size_t i = Size; i < (size_t)_LAYOUT; ++i) Data[i] = 0; Size = _LAYOUT; }}
        unsigned r = Seed >> 2;
        int s = r % _NSLOT; r /= _NSLOT;
        int o = _SOFF[s], w = _SW[s], k = _SK[s];
        if (o + w <= (int)Size) {{
            if (k == 0) {{ long long v = _IVI[r % _NIVI]; memcpy(Data + o, &v, w); }}
            else if (k == 1) {{ float v = _IVF[r % _NIVF]; memcpy(Data + o, &v, 4); }}
            else {{ double v = _IVD[r % _NIVD]; memcpy(Data + o, &v, 8); }}
        }}
        return Size;
    }}
    return LLVMFuzzerMutate(Data, Size, MaxSize);
}}
"""


# in-process libFuzzer differential target

def _return_compare(sig: Signature, capture: bool) -> str:
    if capture:
        return ("  if (_rc <= 0x100000ULL && _ro <= 0x100000ULL && _rc != _ro) __builtin_trap();")
    if sig.is_void_return:
        return ""
    t = sig.return_type.replace(" ", "")
    if t.endswith("*") or t in ("char*", "constchar*"):
        return ""   # pointer returns are compared by content in the replay gate
    if sig.return_type in SCALAR_FLOATS:
        return ("  { double a=(double)_rc, b=(double)_ro; double tol=1e-6*(1.0+(b<0?-b:b));"
                " if (!(a==b) && !(a!=a && b!=b) && (a-b>tol || b-a>tol)) __builtin_trap(); }")
    return "  if (_rc != _ro) __builtin_trap();"


def render_diff_target(sig: Signature, *, typed_decode: bool = True, constants=None) -> str:
    """The in-process libFuzzer target. Decodes once, runs candidate and original, traps on
    divergence. The decode matches the arm. When typed_decode is on and the layout is fixed, a
    custom mutator is appended so the fuzzer places boundary and original constants into the
    correct argument slots."""
    declared, capture = _ret_decl(sig)
    forward = ", ".join(a.c_type for a in sig.args) or "void"
    setup, call_args, buffers = (build_arg_setup(sig) if typed_decode
                                 else build_arg_setup_raw(sig))
    buf_names = {c for c, _b, _base in buffers}

    copies = []
    for call_name, buf, base in buffers:
        copies.append(f"    {base} {buf}_o[sizeof({buf})/sizeof({buf}[0])];")
        copies.append(f"    memcpy({buf}_o, {buf}, sizeof {buf});")
        copies.append(f"    {base} *{call_name}_o = {call_name} ? {buf}_o : ({base}*)0;")
    orig_args = ", ".join(f"{c}_o" if c in buf_names else c for c in call_args)
    cand_args = ", ".join(call_args)

    if capture or not sig.is_void_return:
        call_c = f"    {declared} _rc = cand_func0({cand_args});"
        call_o = f"    {declared} _ro = orig_func0({orig_args});"
    else:
        call_c, call_o = f"    cand_func0({cand_args});", f"    orig_func0({orig_args});"

    cmp_lines = [_return_compare(sig, capture)]
    for _name, buf, _base in buffers:
        cmp_lines.append(f"    if (memcmp({buf}, {buf}_o, sizeof {buf})) __builtin_trap();")
    cmp = "\n".join(l for l in cmp_lines if l)
    mutator = _mutator_c(sig, constants) if typed_decode else ""

    return f"""/* Auto-generated libFuzzer differential target. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
{_TYPEDEFS}
extern {declared} cand_func0({forward});
extern {declared} orig_func0({forward});

{_TAKE}
int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {{
    _len = Size < sizeof(_buf) ? Size : sizeof(_buf);
    memcpy(_buf, Data, _len); _cur = 0;
{chr(10).join(setup)}
{chr(10).join(copies)}
{call_c}
{call_o}
{cmp}
    return 0;
}}
{mutator}
"""


def render_cov_target(sig: Signature, *, typed_decode: bool = True) -> str:
    """Coverage-only target that decodes and calls the candidate alone. Used when the original
    object cannot be linked in-process."""
    declared, capture = _ret_decl(sig)
    forward = ", ".join(a.c_type for a in sig.args) or "void"
    setup, call_args, _ = (build_arg_setup(sig) if typed_decode else build_arg_setup_raw(sig))
    call = ", ".join(call_args)
    invoke = (f"    volatile {declared} _r = cand_func0({call}); (void)_r;"
              if capture or not sig.is_void_return else f"    cand_func0({call});")
    return f"""/* Auto-generated libFuzzer coverage target. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
{_TYPEDEFS}
extern {declared} cand_func0({forward});

{_TAKE}
int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {{
    _len = Size < sizeof(_buf) ? Size : sizeof(_buf);
    memcpy(_buf, Data, _len); _cur = 0;
{chr(10).join(setup)}
{invoke}
    return 0;
}}
"""


# Spellings with an 8-byte sizeof that the typed tables do not know. The raw path reads true
# sizeof, so these must be sized 8 here even though typed decode takes them as 4.
_RAW_SIZEOF_64 = {"ulong", "undefined8", "intptr_t", "uintptr_t", "intmax_t", "uintmax_t"}


def _raw_width(ctype: str) -> int:
    """The true sizeof of ctype on x86-64 LP64, which is what raw decode reads. Unlike the typed
    tables this must be exact for every spelling, since undersizing leaves the argument's high
    bytes unreachable zeros for the whole run."""
    t = norm_ctype(ctype)
    if t.endswith("*"):
        return 8
    if t == "long double":
        return 16
    if t == "double" or t in SCALAR_INTS_64 or t in _RAW_SIZEOF_64:
        return 8
    if t in ("char", "unsigned char", "int8_t", "uint8_t", "bool", "_Bool",
             "uchar", "byte", "undefined", "undefined1"):
        return 1
    if t in ("short", "unsigned short", "int16_t", "uint16_t", "ushort", "undefined2"):
        return 2
    return 4


def max_input_len(sig: Signature, *, typed: bool = True) -> int:
    """Bytes enough to fill every argument, bounded by the input buffer. The two decode paths
    consume very differently, so they are sized separately. Typed decode fills a CAP-element
    buffer per pointer argument and needs hundreds of bytes. Raw decode reads only sizeof(arg)
    bytes per argument, so its bound must be tight or libFuzzer mutates bytes nothing reads."""
    if sig is None or not sig.args:
        return 64
    need = 0
    for a in sig.args:
        ct = a.c_type.replace("const ", "").strip()
        if not typed:
            need += _raw_width(ct)
            continue
        if a.is_string:
            need += STR_BUF + 1
        elif a.is_array:
            base = ct.rstrip("*").strip()
            need += CAP * (8 if base in ("double", "long double") or base in SCALAR_INTS_64
                           else 4) + 1
        elif ct in ("double", "long double") or ct in SCALAR_INTS_64:
            need += 8
        else:
            need += 4
    # the raw path gets no 64-byte floor, its bound is exactly what the harness reads
    return min(INPUT_BUFFER, need if not typed else max(64, need))


# seed corpus, signature-typed inputs that mirror build_arg_setup's consumption

# Boundary integer values. Scalars are packed as signed 32-bit, so INT_MAX and INT_MIN are the
# ceiling.
_INT_BOUNDARIES = [0, 1, -1, 2, -2, 3, 7, 8, 9, 10, 15, 16, 31, 32, 63, 64, 100, 127, 128,
                   255, 256, 1000, 1024, 32767, 32768, 65535, 65536, 1048576, 16777216,
                   0x40000000, 0x7ffffffe, 0x7fffffff,
                   -128, -129, -32768, -65536, -1048576, -0x40000000, -0x7fffffff, -0x80000000]
_FLOAT_BOUNDARIES = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 100.0, -100.0, 1e-9, 1e9,
                     float("inf"), float("-inf"), float("nan")]
_EDGE_SEEDS = [b"", b"\x00", b"\xff" * 16, b"\x00" * 64, b"A" * 1024,
               struct.pack("<i", 0), struct.pack("<i", 1), struct.pack("<i", -1),
               struct.pack("<i", 0x7fffffff), struct.pack("<i", -0x80000000),
               struct.pack("<d", 0.0), struct.pack("<d", 1.0),
               struct.pack("<d", float("inf")), struct.pack("<d", float("nan"))]


# 64-bit boundaries. The int32 ladder plus values straddling the 2^31 and 2^32 lines and the
# int64 ends.
_INT64_BOUNDARIES = _INT_BOUNDARIES + [0x80000000, -0x80000001, 0x100000000, -0x100000000,
                                       0xffffffff, 0x7fffffffffffffff, -0x8000000000000000]


def _draw_int(rng, bias):
    if rng.random() < bias:
        return rng.choice(_INT_BOUNDARIES)
    return rng.choice([0, 1, -1, rng.randint(-1000, 1000), 0x7fffffff, -0x80000000])


def _draw_int64(rng, bias):
    if rng.random() < bias:
        return rng.choice(_INT64_BOUNDARIES)
    return rng.choice([0, 1, -1, rng.randint(-1000, 1000), 0x7fffffff, 0x100000000, -0x80000000])


def _draw_float(rng, bias):
    return rng.choice(_FLOAT_BOUNDARIES) if rng.random() < bias else rng.uniform(-1e3, 1e3)


def _draw_float_ic(rng):
    """A float from the suite's distribution, finite and in range."""
    return rng.choice([0.0, 1.0, -1.0, 0.5]) if rng.random() < 0.15 else rng.uniform(-1000.0, 1000.0)


def _pack_scalar(base, rng, bias, in_contract=False, idx_cap=None) -> bytes:
    if base == "float":
        return struct.pack("<f", _draw_float_ic(rng) if in_contract else _draw_float(rng, bias))
    if base in ("double", "long double"):
        return struct.pack("<d", _draw_float_ic(rng) if in_contract else _draw_float(rng, bias))
    if base in ("char", "int8_t", "uint8_t"):
        return bytes([rng.randint(0, 255)])
    if base in SCALAR_INTS_64:
        if in_contract:
            lo, hi = (0, idx_cap) if idx_cap is not None else (-10000, 10000)
            v = rng.randint(lo, hi)
        else:
            v = max(-(1 << 63), min((1 << 63) - 1, _draw_int64(rng, bias)))
        return struct.pack("<q", v)
    if in_contract:           # index and size scalars stay within the buffer, other ints in range
        lo, hi = (0, idx_cap) if idx_cap is not None else (-10000, 10000)
        return struct.pack("<i", rng.randint(lo, hi))
    return struct.pack("<i", max(-0x80000000, min(0x7fffffff, _draw_int(rng, bias))))


def _distinct_scalar(base, i, in_contract=False, salt=0) -> bytes:
    """A position-unique nonzero value for array element i, the same byte width as _pack_scalar.
    When in_contract, the values stay within the suite's range. The salt varies the fill between
    samples so all-array signatures do not collapse to near-identical seeds, while staying
    constant within one sample to preserve position uniqueness."""
    if base == "float":
        return struct.pack("<f", float(i) * 3.0 + 1.0 + salt)
    if base in ("double", "long double"):
        return struct.pack("<d", float(i) * 3.0 + 1.0 + salt)
    if base in ("char", "int8_t", "uint8_t"):
        return bytes([(i * 37 + 1 + salt) & 0xff])
    mult = 101 if in_contract else 16777619
    if base in SCALAR_INTS_64:
        return struct.pack("<q", (i + 1) * mult + salt)
    # For int32 the fill stays in range for CAP elements, and salt is small enough not to collide
    # with the next element's value, so the fill remains strictly increasing in i.
    return struct.pack("<i", (i + 1) * mult + salt)


def _typed_seed(sig: Signature, rng, bias, distinct: bool = False, in_contract: bool = False,
                salt: int = 0) -> bytes:
    # under in_contract, scalar integer args are capped to the buffer and array contents stay in range
    idx_cap =(CAP - 1) if (in_contract and any(a.is_array and not a.is_string for a in sig.args)) else None
    parts = []
    for arg in sig.args:
        ctype = arg.c_type.replace("const ", "").strip()
        if arg.is_array and not arg.is_string:
            base = ctype.rstrip("*").strip()
            if base not in SCALAR_FLOATS and base not in SCALAR_INTS:
                base = "int"
            for i in range(CAP):                                 # array contents are data, not indices
                parts.append(_distinct_scalar(base, i, in_contract=in_contract, salt=salt)
                             if distinct
                             else _pack_scalar(base, rng, bias, in_contract=in_contract))
            parts.append(bytes([rng.randint(0, 255)]))
        elif arg.is_string:
            length = rng.choice([0, 1, 5, 20])
            parts.append(bytes(rng.randint(33, 126) for _ in range(length)))
            parts.append(b"\x00")
            parts.append(bytes([rng.randint(0, 255)]))
        else:                                                    # scalar arg, capped in contract
            parts.append(_pack_scalar(ctype, rng, bias, in_contract=in_contract, idx_cap=idx_cap))
    return b"".join(parts)


def dedup_seeds(seeds: list) -> list:
    """Order-preserving dedup where the first occurrence wins."""
    seen, out = set(), []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def fresh_batch(sig, *, typed_decode: bool, n: int, seed: int, in_contract: bool) -> list:
    """The fresh evidence batch for a decode regime. This is the one place that maps typed or raw
    onto a corpus generator, so the oracle and retention cannot drift onto different regimes. A
    None sig is deliberately not handled here because the two callers want different things from
    it, so each decides explicitly."""
    if typed_decode:
        return seed_corpus(sig, n=n, seed=seed, in_contract=in_contract)
    return raw_seed_corpus(sig, n=n, seed=seed)


def whole_input_space(sig, seeds: list) -> list:
    """A zero-argument function's whole input space is the one empty invocation. An empty replay
    set must still invoke it once rather than silently pass or leave it unjudged."""
    if not seeds and sig is not None and not sig.args:
        return [b""]
    return seeds


def seed_corpus(sig, *, n: int = 200, bias: float = 0.25, seed: int = 0xC0DE, in_contract: bool = False) -> list:
    """Signature-typed random samples plus generic edge cases, deduplicated. Half the samples use
    position-unique array fills and the other half boundary-biased random values. When in_contract
    is set, every value is drawn from the suite's domain and the extreme edge seeds are dropped."""
    corpus = []
    if sig is not None and sig.args:
        rng = random.Random(seed)
        for i in range(n):
            corpus.append(_typed_seed(sig, rng, bias, distinct=(i % 2 == 1),
                                      in_contract=in_contract, salt=i))
    if not in_contract:
        corpus.extend(_EDGE_SEEDS)   # extreme edge patterns, only outside in-contract mode
    seen, out = set(), []
    for s in corpus:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


def raw_seed_corpus(sig, *, n: int = 200, seed: int = 0xC0DE) -> list:
    """Random raw-byte inputs plus generic edge cases for the raw fuzzer arm, deduplicated."""
    length = max_input_len(sig, typed=False) if sig is not None else 64
    rng = random.Random(seed)
    corpus = [bytes(rng.getrandbits(8) for _ in range(rng.randint(0, length))) for _ in range(n)]
    corpus.extend(_EDGE_SEEDS)
    seen, out = set(), []
    for s in corpus:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


# host-side decoder, mirrors build_arg_setup for feedback display only

def _c_rem(v: int, m: int) -> int:
    q = abs(v) // m
    return v - (q if v >= 0 else -q) * m


def decode_call(sig, seed: bytes, max_chars: int = 300) -> str:
    """Render the func0 call a seed produces, for feedback. Mirrors build_arg_setup's slicing."""
    if sig is None or not sig.args:
        return "func0()"
    cur = 0
    has_buffer = any(a.is_array or a.is_string for a in sig.args)

    def take(n):
        nonlocal cur
        if cur + n > len(seed):
            return None
        b = seed[cur:cur + n]; cur += n
        return b

    def scalar(base):
        if base in SCALAR_FLOATS:
            b = take(4 if base == "float" else 8)
            if b is None:
                return "0"
            return f"{struct.unpack('<f' if base == 'float' else '<d', b)[0]:.6g}"
        if base in ("char", "int8_t"):
            b = take(1)
            v = b[0] if b else 0
            return str(v - 256 if v > 127 else v)
        if base == "uint8_t":
            b = take(1)
            return str(b[0] if b else 0)
        b = take(4)
        return str(int.from_bytes(b, "little", signed=True) if b else 0)

    parts = []
    for arg in sig.args:
        ctype = arg.c_type.replace("const ", "").strip()
        if arg.is_array and not arg.is_string:
            base = ctype.rstrip("*").strip()
            if base not in SCALAR_FLOATS and base not in SCALAR_INTS:
                base = "int"
            elems = [scalar(base) for _ in range(CAP)]
            ctl = take(1)
            if ctl is not None and ctl[0] < 64:
                parts.append(f"{arg.name}=NULL")
            else:
                tail = ", ..." if CAP > DISPLAY_ELEMS else ""
                parts.append(f"{arg.name}=[{', '.join(elems[:DISPLAY_ELEMS])}{tail}]")
        elif arg.is_string:
            chars = bytearray()
            for _ in range(STR_BUF - 1):
                b = take(1)
                if b is None or b[0] == 0:
                    break
                chars.append(b[0])
            ctl = take(1)
            parts.append(f"{arg.name}=NULL" if ctl is not None and ctl[0] < 64
                         else f"{arg.name}={chars.decode('latin-1')!r}")
        else:
            disp = scalar(ctype)
            if has_buffer and ctype in SCALAR_INTS:
                try:
                    disp = str(_c_rem(int(disp), OOC_BOUND + 1))
                except ValueError:
                    pass
            parts.append(f"{arg.name}={disp}")
    call = f"func0({', '.join(parts)})"
    return call if len(call) <= max_chars else call[:max_chars - 4] + "...)"
