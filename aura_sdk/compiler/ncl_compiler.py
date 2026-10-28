"""
aura_sdk/compiler/ncl_compiler.py
───────────────────────────────────
Language: Python
Why: The compiler is a one-time tool that runs offline to produce
     gate netlists. It does not run in the inference hot path.
     Python's expressiveness is the right fit here.

Compiles NCL expression strings to simulatable gate networks:
    compile("AND(OR(a, b), NOT(c))")  →  CompileResult
    cr.print_truth_table()
    cr.run(a=True, b=False, c=False)

Supported ops:
    AND, OR, NOT, XOR, MAJ, AND3, OR3, MUX, THRESHOLD(m,n,...)
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Union

from ..ncl.gates import (
    DualRail, RailState,
    TH22, TH12, TH13, TH23, TH33, THmn,
    NCLInverter,
)
from ..ncl.simulator import NCLSimulator


# ── AST nodes ──────────────────────────────────────────────────────────────

@dataclass
class InputNode:
    name: str

@dataclass
class GateNode:
    op: str
    children: list
    m: int = 0
    n: int = 0

ASTNode = Union[InputNode, GateNode]


# ── Parser ─────────────────────────────────────────────────────────────────

class ExpressionParser:
    OPS = {"AND", "OR", "NOT", "MAJ", "AND3", "OR3", "XOR", "MUX", "THRESHOLD"}

    def __init__(self, source: str):
        self.source = source.strip()
        self.pos    = 0

    def parse(self) -> ASTNode:
        node = self._expr()
        self._skip_ws()
        if self.pos < len(self.source):
            raise SyntaxError(
                f"Unexpected token at pos {self.pos}: "
                f"'{self.source[self.pos:self.pos+10]}'"
            )
        return node

    def _skip_ws(self):
        while self.pos < len(self.source) and self.source[self.pos] in " \t\n":
            self.pos += 1

    def _expr(self) -> ASTNode:
        self._skip_ws()
        for op in self.OPS:
            if self.source[self.pos:self.pos+len(op)] == op:
                rest = self.source[self.pos+len(op):]
                if rest and rest[0] == "(":
                    return self._gate(op)
        return self._input()

    def _gate(self, op: str) -> GateNode:
        self.pos += len(op)
        self._skip_ws()
        assert self.source[self.pos] == "("
        self.pos += 1
        children = []
        m = n = 0
        if op == "THRESHOLD":
            m = int(self._read_int()); self._expect(",")
            n = int(self._read_int()); self._expect(",")
        while True:
            self._skip_ws()
            if self.source[self.pos] == ")":
                self.pos += 1; break
            children.append(self._expr())
            self._skip_ws()
            if self.source[self.pos]   == ",":  self.pos += 1
            elif self.source[self.pos] == ")":  self.pos += 1; break
            else: raise SyntaxError(f"Expected ',' or ')' at pos {self.pos}")
        return GateNode(op=op, children=children, m=m, n=n)

    def _input(self) -> InputNode:
        self._skip_ws()
        match = re.match(r"[a-zA-Z_][a-zA-Z0-9_]*", self.source[self.pos:])
        if not match:
            raise SyntaxError(f"Expected identifier at pos {self.pos}")
        name = match.group(0)
        self.pos += len(name)
        return InputNode(name=name)

    def _read_int(self) -> str:
        self._skip_ws()
        match = re.match(r"\d+", self.source[self.pos:])
        if not match: raise SyntaxError(f"Expected integer at pos {self.pos}")
        val = match.group(0); self.pos += len(val); return val

    def _expect(self, char: str):
        self._skip_ws()
        if self.source[self.pos] != char:
            raise SyntaxError(f"Expected '{char}' at pos {self.pos}")
        self.pos += 1


# ── Synthesiser ────────────────────────────────────────────────────────────

class NCLSynthesiser:
    def __init__(self):
        self.sim    = NCLSimulator()
        self.inputs: dict[str, DualRail] = {}
        self._count = 0

    def _uid(self, prefix: str = "g") -> str:
        self._count += 1
        return f"{prefix}_{self._count}"

    def _get_input(self, name: str) -> DualRail:
        if name not in self.inputs:
            self.inputs[name] = self.sim.input(name)
        return self.inputs[name]

    def synthesise(self, node: ASTNode) -> DualRail:
        if isinstance(node, InputNode):
            return self._get_input(node.name)

        op  = node.op
        ch  = [self.synthesise(c) for c in node.children]
        uid = self._uid(op.lower())

        if op == "AND":
            assert len(ch) == 2
            return self.sim.add_gate(TH22(uid), ch, f"{uid}_out")
        elif op == "OR":
            assert len(ch) == 2
            return self.sim.add_gate(TH12(uid), ch, f"{uid}_out")
        elif op == "NOT":
            assert len(ch) == 1
            return self.sim.add_gate(NCLInverter(uid), ch, f"{uid}_out")
        elif op == "AND3":
            assert len(ch) == 3
            return self.sim.add_gate(TH33(uid), ch, f"{uid}_out")
        elif op == "OR3":
            assert len(ch) == 3
            return self.sim.add_gate(TH13(uid), ch, f"{uid}_out")
        elif op == "MAJ":
            assert len(ch) == 3
            return self.sim.add_gate(TH23(uid), ch, f"{uid}_out")
        elif op == "XOR":
            assert len(ch) == 2
            a, b = ch
            na = self.sim.add_gate(NCLInverter(f"{uid}_na"), [a], f"{uid}_na")
            nb = self.sim.add_gate(NCLInverter(f"{uid}_nb"), [b], f"{uid}_nb")
            t0 = self.sim.add_gate(TH22(f"{uid}_t0"), [a, nb],  f"{uid}_t0")
            t1 = self.sim.add_gate(TH22(f"{uid}_t1"), [na, b],  f"{uid}_t1")
            return self.sim.add_gate(TH12(uid), [t0, t1], f"{uid}_out")
        elif op == "MUX":
            assert len(ch) == 3
            sel, d0, d1 = ch
            inv_sel = self.sim.add_gate(NCLInverter(f"{uid}_inv"), [sel], f"{uid}_inv")
            b0 = self.sim.add_gate(TH22(f"{uid}_b0"), [d0, inv_sel], f"{uid}_b0")
            b1 = self.sim.add_gate(TH22(f"{uid}_b1"), [d1, sel],     f"{uid}_b1")
            return self.sim.add_gate(TH12(uid), [b0, b1], f"{uid}_out")
        elif op == "THRESHOLD":
            m, n = node.m, node.n
            assert len(ch) == n
            return self.sim.add_gate(THmn(m, n, uid), ch, f"{uid}_out")
        else:
            raise ValueError(f"Unknown op: {op}")


# ── Public API ─────────────────────────────────────────────────────────────

@dataclass
class CompileResult:
    source:      str
    input_names: list[str]
    output_rail: DualRail
    sim:         NCLSimulator
    gate_count:  int
    stats:       dict = field(default_factory=dict)

    def run(self, **input_values: bool):
        self.sim.reset()
        self.sim.inject(input_values)
        return self.sim.run()

    def truth_table(self) -> list[dict]:
        n = len(self.input_names)
        if n > 12:
            raise ValueError(f"Truth table with {n} inputs too large (max 12)")
        rows = []
        for i in range(2 ** n):
            vals = {}
            for j, name in enumerate(self.input_names):
                vals[name] = bool((i >> (n - 1 - j)) & 1)
            result = self.run(**vals)
            out    = result.output_values.get(self.output_rail.name)
            rows.append({**vals, "output": out,
                         "energy_pJ":  result.total_energy_pJ,
                         "latency_ns": result.latency_ns})
        return rows

    def print_truth_table(self):
        rows = self.truth_table()
        if not rows: print("(empty)"); return
        headers = self.input_names + ["output", "energy_pJ", "latency_ns"]
        cw = max(len(h) for h in headers) + 2
        print(f"\n  Circuit : {self.source}")
        print(f"  Gates   : {self.gate_count}")
        print(f"  Idle    : 0.00 W\n")
        print("  " + "".join(h.ljust(cw) for h in headers))
        print("  " + "─" * (cw * len(headers)))
        for row in rows:
            line = ""
            for h in headers:
                v = row.get(h)
                if isinstance(v, float): line += f"{v:.3f}".ljust(cw)
                elif isinstance(v, bool): line += ("1" if v else "0").ljust(cw)
                else: line += str(v).ljust(cw)
            print("  " + line)
        print()

    def __repr__(self):
        return (f"CompileResult(source='{self.source}', "
                f"inputs={self.input_names}, gates={self.gate_count})")


def compile(source: str, output_name: str = "out") -> CompileResult:
    """
    Compile an NCL expression to a simulatable gate network.

    Parameters
    ──────────
    source      : NCL expression, e.g. 'AND(OR(a, b), NOT(c))'
    output_name : Name to give the output rail (default 'out')

    Returns
    ───────
    CompileResult

    Example
    ───────
    >>> cr = compile("AND(OR(a, b), NOT(c))")
    >>> cr.print_truth_table()
    >>> r = cr.run(a=True, b=False, c=False)
    >>> print(r.output_values)
    """
    parser = ExpressionParser(source)
    ast    = parser.parse()
    synth  = NCLSynthesiser()
    out    = synth.synthesise(ast)
    out.name = output_name
    synth.sim.output(output_name, out)
    return CompileResult(
        source      = source,
        input_names = list(synth.inputs.keys()),
        output_rail = out,
        sim         = synth.sim,
        gate_count  = synth._count,
        stats       = {"idle_power_W": 0.0},
    )
