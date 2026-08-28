"""Recover func0's signature from C source. Parsing tolerates Ghidra pseudo-types and K&R
star placement."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Buffer sizes the harness and the seed packer agree on. A pointer or array argument is a
# CAP-element buffer, a loose integer is bounded in magnitude by OOC_BOUND, and a string is a
# STR_BUF-byte NUL-terminated buffer.
CAP = 64
STR_BUF = 256
OOC_BOUND = 1024
INPUT_BUFFER = 4096
DISPLAY_ELEMS = 8       # array elements shown in feedback, the rest are consumed not shown

SCALAR_INTS = {"int", "short", "char", "long", "long long", "size_t", "ssize_t",
               "ptrdiff_t", "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t",
               "uint16_t", "uint32_t", "uint64_t", "unsigned", "unsigned int",
               "signed", "signed int"}
# Integer types decoded and packed at full 8-byte width so values outside the 32-bit range are
# reachable. Members are canonical spellings, see norm_ctype.
SCALAR_INTS_64 = {"long", "long long", "size_t", "ssize_t", "ptrdiff_t", "int64_t", "uint64_t",
                  "unsigned long", "unsigned long long"}
SCALAR_FLOATS = {"float", "double", "long double"}

_TYPE_WORDS = frozenset({
    "void", "int", "char", "long", "short", "float", "double", "bool", "unsigned",
    "signed", "const", "static", "size_t", "ssize_t", "ptrdiff_t", "int8_t", "int16_t",
    "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "uint", "ulong",
    "ushort", "uchar", "byte", "undefined", "undefined1", "undefined2", "undefined4",
    "undefined8"})

# The return type ends in whitespace or a star so both "long func0(" and "char *func0(" match.
_SIGNATURE_RE = re.compile(r"(?P<ret>[A-Za-z_][\w\s\*]*[\s\*])func0\s*\((?P<args>[^)]*)\)",
                           re.MULTILINE)

# Ghidra never emits the name func0, so the pseudo C prototype has to be matched by shape rather
# than by name. The trailing brace requires a definition, which keeps calls and forward
# declarations out, and callers additionally require a plausible return type.
_ANY_SIGNATURE_RE = re.compile(
    r"(?P<ret>[A-Za-z_][\w\s\*]*[\s\*])(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^)]*)\)\s*\{",
    re.MULTILINE)

# Leading storage and inline qualifiers on func0's definition only, never on helpers.
_FUNC0_QUALS = re.compile(
    r"\b(?:(?:static|inline|__inline__|__inline|__forceinline|register)\s+)+"
    r"(?=[\w\s\*]*\bfunc0\s*\()")


@dataclass
class Arg:
    c_type: str            # e.g. "int", "float*", "char*"
    name: str
    is_array: bool = False
    is_string: bool = False


@dataclass
class Signature:
    return_type: str
    args: list = field(default_factory=list)
    is_void_return: bool = False


def externalize_func0(src: str) -> str:
    """Remove leading static/inline qualifiers from func0 so it has external linkage."""
    return _FUNC0_QUALS.sub("", src, count=1)


def is_pointer(arg: Arg) -> bool:
    return bool(arg.is_array or arg.is_string or arg.c_type.replace(" ", "").endswith("*"))


# Qualifiers that never change a type's byte layout.
_LAYOUT_QUALS = ("const", "volatile", "register", "signed")


def norm_ctype(ctype: str) -> str:
    """Canonical spelling of a C type. Layout-neutral qualifiers are dropped, whitespace is
    collapsed, int suffixes are folded, and stars are attached. Applied once at parse time so
    every downstream spelling table sees only canonical forms."""
    t = " ".join(w for w in ctype.replace("*", " * ").split() if w not in _LAYOUT_QUALS)
    for long_form, short in (("long long int", "long long"), ("long int", "long"),
                             ("short int", "short")):
        t = t.replace(long_form, short)
    return t.replace(" *", "*")


def layout_class(ctype: str) -> str:
    """The width and kind class the typed decoder consumes for this spelling. This mirrors the
    harness decode tables exactly, quirks included, because sig_key promises that the same key
    decodes a seed into the same call, and the decoder is those tables rather than true sizeof.
    The raw path's true sizeof accounting lives separately in harness._raw_width."""
    t = norm_ctype(ctype)
    depth = t.count("*")
    base = t.replace("*", "").strip()
    if base == "float":
        cls = "f4"
    elif base in ("double", "long double"):
        cls = "f8"
    elif base in ("char", "int8_t", "uint8_t"):
        cls = "i1"
    elif base in SCALAR_INTS_64:
        cls = "i8"
    elif base == "void":
        cls = "v0"
    else:
        cls = "i4"
    return "p" * depth + cls


def sig_key(sig) -> str:
    """Identity of the byte consumption layout a seed was generated under. Two signatures with
    the same key decode a seed into the same call, so counterexamples are interchangeable between
    them. Keyed on each argument's layout class rather than its raw spelling, so a candidate that
    merely re-spells a type cannot escape the accumulated counterexamples."""
    if sig is None:
        return "-"
    parts = [("s" if a.is_string else "a" if a.is_array else "v") + ":" + layout_class(a.c_type)
             for a in sig.args]
    return f"{layout_class(sig.return_type)}({','.join(parts)})"


def incompatibility(cand, ref, *, check_pointers: bool = False) -> str | None:
    """How cand disagrees with the decompiler-recovered prototype ref, else None. Diagnostic
    only, callers record this and do not reject on it. Ghidra routinely reports a pointer
    parameter as long on stripped binaries, so pointer checks stay behind check_pointers, and
    even arity is wrong often enough that acting on it would break correct candidates."""
    if cand is None or ref is None:
        return None
    if len(cand.args) != len(ref.args):
        return (f"arity: candidate takes {len(cand.args)} parameter(s), recovered prototype "
                f"takes {len(ref.args)}")
    if not check_pointers:
        return None
    wrong = [(i, c, r) for i, (c, r) in enumerate(zip(cand.args, ref.args))
             if is_pointer(c) != is_pointer(r)]
    if wrong:
        i, c, r = wrong[0]
        want = "pointer" if is_pointer(r) else "by-value scalar"
        got = "pointer" if is_pointer(c) else "by-value scalar"
        return f"parameter {i + 1}: candidate {got} ({c.c_type}), recovered {want} ({r.c_type})"
    return None


def _plausible_return(ret: str) -> bool:
    words = ret.replace("*", " ").split()
    return bool(words) and all(w in _TYPE_WORDS for w in words)


def _balanced_args(src: str, open_idx: int):
    """Text between the paren at open_idx and its matching close paren, honouring nesting."""
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                return src[open_idx + 1:i]
    return None


def _split_top_level(args: str) -> list:
    """Split a C arg list on top-level commas, keeping function pointer and array declarators
    whole."""
    out, depth, cur = [], 0, []
    for ch in args:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def parse_signature(source: str):
    """Extract func0's signature from pseudo C or candidate C, or None if absent. The first
    match with a plausible return type wins."""
    match = None
    for m in _SIGNATURE_RE.finditer(source):
        if _plausible_return(m.group("ret").strip()):
            match = m
            break
        if match is None:
            match = m
    if not match:
        return None
    return _build(source, match)


def parse_any_signature(source: str):
    """The signature of the first function defined in source, whatever it is called. This is how
    Ghidra's recovered prototype is read, since the pseudo C never names the function func0.
    Unlike parse_signature there is no fallback to an implausible return type, because a wrong
    guess is worse than no answer here."""
    for m in _ANY_SIGNATURE_RE.finditer(source):
        if _plausible_return(m.group("ret").strip()):
            return _build(source, m)
    return None


def _build(source: str, match):
    """Turn a matched prototype into a Signature. Shared by both parsers. Types are stored in
    canonical spelling so the decode tables never see a variant form."""
    return_type = norm_ctype(match.group("ret").strip())
    open_idx = match.start() + match.group(0).index("(")
    args_raw = _balanced_args(source, open_idx)
    if args_raw is None:
        args_raw = match.group("args")
    args_raw = args_raw.strip()

    sig = Signature(return_type=return_type, is_void_return=(return_type == "void"))
    if args_raw in ("", "void"):
        return sig

    for raw in _split_top_level(args_raw):
        raw = raw.strip()
        if not raw:
            continue
        # keep a declarator with parentheses whole, such as a function pointer
        if "(" in raw:
            sig.args.append(Arg(c_type=re.sub(r"\s+", " ", raw), name=f"arg{len(sig.args)}"))
            continue
        is_array = "[" in raw
        cleaned = raw.split("[", 1)[0].strip()
        parts = cleaned.rsplit(None, 1)
        if len(parts) == 2:
            ctype, name = parts
        elif "*" in cleaned:               # unspaced declarator, the name is after the last star
            star = cleaned.rfind("*")
            ctype, name = cleaned[:star + 1], cleaned[star + 1:].strip() or f"arg{len(sig.args)}"
        else:
            ctype, name = cleaned, f"arg{len(sig.args)}"
        stars = name.count("*")            # stars bind to the declarator, so move them onto the type
        name = name.lstrip("*").strip()
        ctype = norm_ctype(ctype.strip() + "*" * stars)
        if is_array and "*" not in ctype:
            ctype += "*"
        is_string = ctype == "char*"       # norm_ctype already folded const char variants into this
        sig.args.append(Arg(c_type=ctype, name=name,
                            is_array=is_array or (ctype.endswith("*") and not is_string),
                            is_string=is_string))
    return sig
