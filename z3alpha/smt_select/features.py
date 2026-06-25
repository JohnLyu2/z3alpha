"""
SMT-LIB benchmark feature extractor.

This is a Python re-implementation of Klammerhammer (klhm) in Zig, a tool for extracting
metadata from SMT-LIB 2 benchmark files.

Original implementation: klhm by Hans-Joerg Schurr <commits@schurr.at>
Source: https://github.com/SMT-LIB/SMT-LIB-db/tree/main/klhm
License: BSD 3-Clause

Python re-implementation by Claude Code.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─── SMT-LIB symbol table ────────────────────────────────────────────────────
# Matches klhm/src/smtlib-symbols exactly (including the duplicate "-" which
# keeps the array at 204 entries to match klhm's StaticStringMap kvs.len).

_SYMBOLS_RAW = """\
true
false
Bool
ite
not
or
and
=>
xor
=
distinct
const
forall
exists
let
Int
div
mod
divisible
iand
int.pow2
div_total
mod_total
Real
/
/_total
+
-
*
<
<=
>
>=
to_real
to_int
is_int
abs
^
-
real.pi
exp
sin
cos
tan
csc
sec
cot
arcsin
arccos
arctan
arccsc
arcsec
arccot
sqrt
BitVec
bvempty
concat
extract
repeat
bvnot
bvand
bvor
bvnand
bvnor
bvxor
bvxnor
bvcomp
bvneg
bvadd
bvmul
bvudiv
bvurem
bvsub
bvsdiv
bvsrem
bvsmod
bvult
bvule
bvugt
bvuge
bvslt
bvsle
bvsgt
bvsge
bvshl
bvlshr
bvashr
zero_extend
sign_extend
rotate_left
rotate_right
reduce_and
reduce_or
reduce_xor
bvite
bv1ult
bitOf
bvuaddo
bvsaddo
bvumulo
bvsmulo
bvusubo
bvssubo
bvsdivo
bvultbv
bvsltbv
bvredand
bvredor
int2bv
bv2nat
bvsize
Array
select
store
store_all
eqrange
FloatingPoint
RoundingMode
fp
fp.add
fp.sub
fp.mul
fp.div
fp.fma
fp.sqrt
fp.rem
fp.roundToIntegral
fp.min
fp.max
fp.abs
fp.neg
fp.leq
fp.lt
fp.geq
fp.gt
fp.eq
fp.isNormal
fp.isSubnormal
fp.isZero
fp.isInfinite
fp.isNaN
fp.isPositive
fp.isNegative
roundNearestTiesToEven
roundNearestTiesToAway
roundTowardPositive
roundTowardNegative
roundTowardZero
fp.to_ubv
fp.to_ubv_total
fp.to_sbv
fp.to_sbv_total
fp.to_real
to_fp
to_fp_unsigned
to_fp_bv
String
Char
RegLan
str.len
str.++
str.substr
str.contains
str.replace
str.indexof
str.at
str.prefixof
str.suffixof
str.rev
str.unit
str.update
str.to_lower
str.to_upper
str.to_code
str.from_code
str.is_digit
str.to_int
str.from_int
str.<
str.<=
str.replace_all
str.replace_re
str.replace_re_all
str.indexof_re
re.allchar
re.none
re.all
re.empty
str.to_re
re.*
re.+
re.opt
re.comp
re.range
re.++
re.inter
re.union
re.diff
re.loop
str.in_re
seq.empty
seq.unit
seq.nth
seq.len"""

SMTLIB_SYMBOLS: list[str] = _SYMBOLS_RAW.splitlines()
NUM_SYMBOLS: int = len(SMTLIB_SYMBOLS)  # 204

# Map symbol string → index (first occurrence wins, matching Zig StaticStringMap)
_SYMBOL_INDEX: dict[str, int] = {}
for _i, _s in enumerate(SMTLIB_SYMBOLS):
    _SYMBOL_INDEX.setdefault(_s, _i)


# ─── Tokenizer ───────────────────────────────────────────────────────────────

_SYMBOL_CHARS = frozenset(
    "~!@$%^&*_-+=<>.:?/#"
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

_WHITESPACE = frozenset("\t\n\r ")

# Token type constants
_TOK_OPEN = "("
_TOK_CLOSE = ")"
_TOK_SYM = "sym"
_TOK_STR = "str"


class _TokenIterator:
    __slots__ = ("_data", "_pos", "_cached")

    def __init__(self, data: str) -> None:
        self._data = data
        self._pos: int = 0
        self._cached: Optional[tuple[str, str]] = None

    @property
    def pos(self) -> int:
        return self._pos

    def next(self) -> Optional[tuple[str, str]]:
        if self._cached is not None:
            tok = self._cached
            self._cached = None
            return tok
        return self._read()

    def peek(self) -> Optional[tuple[str, str]]:
        if self._cached is None:
            self._cached = self._read()
        return self._cached

    def _read(self) -> Optional[tuple[str, str]]:
        data = self._data
        pos = self._pos
        n = len(data)

        in_comment = False
        while pos < n:
            c = data[pos]
            if c == ";":
                in_comment = True
                pos += 1
                continue
            if in_comment:
                if c == "\n":
                    in_comment = False
                pos += 1
                continue
            if c in _WHITESPACE:
                pos += 1
                continue
            break

        if pos >= n:
            self._pos = pos
            return None

        c = data[pos]

        if c == "(":
            self._pos = pos + 1
            return (_TOK_OPEN, "(")

        if c == ")":
            self._pos = pos + 1
            return (_TOK_CLOSE, ")")

        if c == "|":
            pos += 1
            start = pos
            while pos < n and data[pos] != "|":
                pos += 1
            span = data[start:pos]
            self._pos = pos + 1
            return (_TOK_STR, span)

        if c == '"':
            pos += 1
            start = pos
            while pos < n:
                if data[pos] == '"':
                    if pos + 1 < n and data[pos + 1] == '"':
                        pos += 2
                        continue
                    break
                pos += 1
            span = data[start:pos]
            self._pos = pos + 1
            return (_TOK_STR, span)

        start = pos
        while pos < n and data[pos] in _SYMBOL_CHARS:
            pos += 1
        span = data[start:pos]
        self._pos = pos
        return (_TOK_SYM, span)


# ─── Data structures ──────────────────────────────────────────────────────────


@dataclass
class QueryData:
    normalizedSize: int = 0
    compressedSize: int = 0
    assertsCount: int = 0
    declareFunCount: int = 0
    declareConstCount: int = 0
    declareSortCount: int = 0
    defineFunCount: int = 0
    defineFunRecCount: int = 0
    constantFunCount: int = 0
    defineSortCount: int = 0
    declareDatatypeCount: int = 0
    maxTermDepth: int = 0
    status: Optional[str] = None
    symbolFrequency: list[int] = field(default_factory=lambda: [0] * NUM_SYMBOLS)

    def to_dict(self) -> dict:
        return {
            "normalizedSize": self.normalizedSize,
            "assertsCount": self.assertsCount,
            "declareFunCount": self.declareFunCount,
            "declareConstCount": self.declareConstCount,
            "declareSortCount": self.declareSortCount,
            "defineFunCount": self.defineFunCount,
            "defineFunRecCount": self.defineFunRecCount,
            "constantFunCount": self.constantFunCount,
            "defineSortCount": self.defineSortCount,
            "declareDatatypeCount": self.declareDatatypeCount,
            "maxTermDepth": self.maxTermDepth,
            "status": self.status,
            "symbolFrequency": self.symbolFrequency,
        }


@dataclass
class BenchmarkData:
    logic: Optional[str] = None
    size: int = 0
    compressedSize: int = 0
    license: Optional[str] = None
    generatedOn: Optional[str] = None
    generatedBy: Optional[str] = None
    targetSolver: Optional[str] = None
    timeLimit: Optional[str] = None
    generator: Optional[str] = None
    application: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    queryCount: int = 0
    isIncremental: bool = False

    def to_dict(self) -> dict:
        return {
            "logic": self.logic,
            "size": self.size,
            "license": self.license,
            "generatedOn": self.generatedOn,
            "generatedBy": self.generatedBy,
            "targetSolver": self.targetSolver,
            "timeLimit": self.timeLimit,
            "generator": self.generator,
            "application": self.application,
            "description": self.description,
            "category": self.category,
            "queryCount": self.queryCount,
            "isIncremental": self.isIncremental,
        }


# ─── Source field parser (mirrors data.zig set_source) ────────────────────────

_SOURCE_KEYS = {
    "Generated by": "generatedBy",
    "Generated on": "generatedOn",
    "Application": "application",
    "Target solver": "targetSolver",
    "Time limit": "timeLimit",
    "Generator": "generator",
    "Generated By": "generatedBy",
    "Generated On": "generatedOn",
    "Target Solver": "targetSolver",
    "Time Limit": "timeLimit",
}


def _parse_source(bm: BenchmarkData, src: str) -> None:
    lines = src.split("\n")
    i = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            colon = stripped.index(":")
            left = stripped[:colon].strip()
            right = stripped[colon + 1:].strip(" \t\n\r,;:")
            if left in _SOURCE_KEYS:
                setattr(bm, _SOURCE_KEYS[left], right)
                continue
        # Not a known header line — rest is description
        break
    else:
        return
    rest = "\n".join(lines[i:]).strip()
    if rest:
        bm.description = rest


# ─── Parser helpers ───────────────────────────────────────────────────────────


def _skip_rest(it: _TokenIterator) -> int:
    """Skip tokens until the closing ) that ends the current s-expression.
    Assumes the opening ( has already been consumed (starts at level=1).
    Returns position after the closing ).
    """
    level = 1
    while True:
        tok = it.next()
        if tok is None:
            break
        if tok[0] == _TOK_OPEN:
            level += 1
        elif tok[0] == _TOK_CLOSE:
            level -= 1
            if level == 0:
                return it.pos
    return it.pos


def _read_term(it: _TokenIterator, qd: QueryData) -> int:
    """Read one term, updating symbol frequency and max depth.
    At level 0: reads one atom (symbol/string) or one s-expression.
    Returns position after the term.
    """
    level = 0
    while True:
        tok = it.next()
        if tok is None:
            break
        ttype = tok[0]
        if ttype == _TOK_OPEN:
            level += 1
        elif ttype == _TOK_CLOSE:
            if level > qd.maxTermDepth:
                qd.maxTermDepth = level
            level -= 1
            if level == 0:
                return it.pos
        elif ttype == _TOK_SYM:
            idx = _SYMBOL_INDEX.get(tok[1])
            if idx is not None:
                qd.symbolFrequency[idx] += 1
            if level == 0:
                return it.pos
        else:  # _TOK_STR
            if level == 0:
                return it.pos
    return it.pos


def _get_string(it: _TokenIterator) -> Optional[str]:
    tok = it.next()
    if tok is not None and tok[0] == _TOK_STR:
        return tok[1]
    return None


# ─── Main extractor ───────────────────────────────────────────────────────────


def extract_features(path: str | Path) -> list[dict]:
    """Parse an SMT-LIB 2 file and return a list of dicts.

    The list contains one dict per check-sat call (query statistics),
    followed by one dict with overall benchmark metadata — matching
    klhm's JSON array output format.
    """
    content = Path(path).read_text(encoding="utf-8", errors="replace")
    it = _TokenIterator(content)

    bm = BenchmarkData(size=len(content.encode("utf-8")))

    # Each scope: {'intervals': [pos, ...], 'data': QueryData}
    # intervals alternates start/end markers. Starts open: [0]
    scopes: list[dict] = [{"intervals": [0], "data": QueryData()}]

    results: list[dict] = []

    while True:
        tok = it.next()
        if tok is None:
            break
        if tok[0] != _TOK_OPEN:
            continue  # unexpected token at top level, skip

        # position of the '(' we just consumed
        level_start_idx = it.pos - 1

        cmd_tok = it.next()
        if cmd_tok is None:
            break

        top = scopes[-1]

        if cmd_tok[0] != _TOK_SYM:
            _skip_rest(it)
            continue

        cmd = cmd_tok[1]

        if cmd == "set-logic":
            tok2 = it.next()
            if tok2:
                bm.logic = tok2[1]
            _skip_rest(it)

        elif cmd == "set-info":
            attr_tok = it.next()
            if attr_tok and attr_tok[0] == _TOK_SYM:
                attr = attr_tok[1]
                if attr == ":status":
                    v = it.next()
                    if v and v[0] == _TOK_SYM:
                        top["data"].status = v[1]
                elif attr == ":license":
                    v = _get_string(it)
                    if v is not None:
                        bm.license = "CMU SoSy Lab" if len(v) > 150 else v
                elif attr == ":category":
                    v = _get_string(it)
                    if v is not None:
                        bm.category = v
                elif attr == ":source":
                    v = _get_string(it)
                    if v is not None:
                        _parse_source(bm, v)
            _skip_rest(it)

        elif cmd == "assert":
            top["data"].assertsCount += 1
            _read_term(it, top["data"])
            _skip_rest(it)

        elif cmd == "declare-fun":
            it.next()  # name
            it.next()  # '(' of param list
            nxt = it.peek()
            if nxt is None:
                break
            if nxt[0] == _TOK_CLOSE:
                top["data"].declareConstCount += 1
                it.next()  # consume ')'
            else:
                top["data"].declareFunCount += 1
                _skip_rest(it)  # skip param list
            _skip_rest(it)  # skip return sort + outer ')'

        elif cmd == "declare-const":
            top["data"].declareConstCount += 1
            _skip_rest(it)

        elif cmd == "declare-sort":
            top["data"].declareSortCount += 1
            _skip_rest(it)

        elif cmd == "define-fun":
            it.next()  # name
            it.next()  # '(' of param list
            nxt = it.peek()
            if nxt is None:
                break
            if nxt[0] == _TOK_CLOSE:
                top["data"].constantFunCount += 1
                it.next()  # consume ')'
            else:
                top["data"].defineFunCount += 1
                _skip_rest(it)  # skip param list
            _read_term(it, top["data"])  # return sort
            _read_term(it, top["data"])  # body
            _skip_rest(it)  # outer ')'

        elif cmd == "define-fun-rec":
            top["data"].defineFunRecCount += 1
            it.next()  # name
            _read_term(it, top["data"])  # param list
            _read_term(it, top["data"])  # return sort
            _read_term(it, top["data"])  # body
            _skip_rest(it)

        elif cmd == "define-funs-rec":
            it.next()  # '(' of signature list
            count = 0
            while True:
                nxt = it.peek()
                if nxt is None:
                    break
                if nxt[0] == _TOK_CLOSE:
                    it.next()
                    break
                _read_term(it, top["data"])  # one signature
                count += 1
            top["data"].defineFunRecCount += count
            _read_term(it, top["data"])  # definition list
            _skip_rest(it)

        elif cmd == "declare-datatype":
            top["data"].declareDatatypeCount += 1
            _skip_rest(it)

        elif cmd == "declare-datatypes":
            it.next()  # '(' of sort list
            while True:
                nxt = it.peek()
                if nxt is None:
                    break
                if nxt[0] == _TOK_CLOSE:
                    it.next()
                    break
                _read_term(it, top["data"])  # one sort declaration
                top["data"].declareDatatypeCount += 1
            _skip_rest(it)

        elif cmd == "define-sort":
            top["data"].defineSortCount += 1
            _skip_rest(it)

        elif cmd == "push":
            n_tok = it.next()
            if n_tok is None:
                break
            try:
                n = int(n_tok[1])
            except (ValueError, TypeError):
                n = 1

            # Close current interval up to the push command
            old_idx = top["intervals"][-1]
            top["data"].normalizedSize += level_start_idx - old_idx
            top["intervals"].append(level_start_idx)

            idx = _skip_rest(it)

            for _ in range(n):
                new_scope = {
                    "intervals": [],
                    "data": copy.deepcopy(top["data"]),
                }
                scopes.append(new_scope)
                top = scopes[-1]

            top["intervals"].append(idx)

        elif cmd == "pop":
            n_tok = it.next()
            if n_tok is None:
                break
            try:
                n = int(n_tok[1])
            except (ValueError, TypeError):
                n = 1

            top["intervals"].append(level_start_idx)

            idx = _skip_rest(it)

            for _ in range(n):
                if len(scopes) > 1:
                    scopes.pop()
                    top = scopes[-1]

            top["intervals"].append(idx)

        elif cmd in ("check-sat", "check-sat-assuming"):
            old_idx = top["intervals"][-1]
            top["intervals"].append(level_start_idx)
            idx = _skip_rest(it)

            bm.queryCount += 1
            top["data"].normalizedSize += idx - old_idx

            results.append(top["data"].to_dict())
            top["intervals"].append(idx)

        elif cmd == "exit":
            break

        else:
            _skip_rest(it)

    bm.isIncremental = bm.queryCount > 1
    results.append(bm.to_dict())
    return results


def extract_features_json(path: str | Path) -> str:
    """Like extract_features but returns a JSON string."""
    return json.dumps(extract_features(path))
