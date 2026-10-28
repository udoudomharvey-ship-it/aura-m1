"""
aura_sdk.ncl.gates
──────────────────
Null Convention Logic gate primitives for the Aura-M1.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class RailState(Enum):
    NULL    = (0, 0)
    DATA0   = (1, 0)
    DATA1   = (0, 1)
    INVALID = (1, 1)

    @property
    def wire_0(self): return self.value[0]
    @property
    def wire_1(self): return self.value[1]
    @property
    def is_null(self):    return self == RailState.NULL
    @property
    def is_data(self):    return self in (RailState.DATA0, RailState.DATA1)
    @property
    def logic_value(self) -> Optional[bool]:
        if self == RailState.DATA1: return True
        if self == RailState.DATA0: return False
        return None

    @classmethod
    def from_bool(cls, v: bool) -> 'RailState':
        return cls.DATA1 if v else cls.DATA0

    def __repr__(self):
        return {RailState.NULL:'NULL', RailState.DATA0:'DATA0',
                RailState.DATA1:'DATA1', RailState.INVALID:'INVALID'}[self]


@dataclass
class DualRail:
    name: str
    state: RailState = RailState.NULL
    _transitions: int = field(default=0, repr=False)

    def set(self, s: RailState):
        if s != self.state:
            self._transitions += 1
            self.state = s
    def set_null(self):          self.set(RailState.NULL)
    def set_data(self, v: bool): self.set(RailState.from_bool(v))

    @property
    def is_null(self):  return self.state.is_null
    @property
    def is_data(self):  return self.state.is_data
    @property
    def value(self):    return self.state.logic_value
    @property
    def transitions(self): return self._transitions


class NCLGate:
    gate_count = 0
    def __init__(self, name=None):
        NCLGate.gate_count += 1
        self.name = name or f"{self.__class__.__name__}_{NCLGate.gate_count}"
        self._eval_count = 0

    def evaluate(self, *inputs): raise NotImplementedError

    def __call__(self, *inputs):
        r = self.evaluate(*inputs)
        if r != RailState.NULL: self._eval_count += 1
        return r

    @property
    def eval_count(self): return self._eval_count


class TH22(NCLGate):
    def evaluate(self, a, b):
        if a.is_null or b.is_null: return RailState.NULL
        return RailState.DATA1 if (a.value and b.value) else RailState.DATA0

class TH12(NCLGate):
    def evaluate(self, a, b):
        if a.is_null or b.is_null: return RailState.NULL
        return RailState.DATA1 if (a.value or b.value) else RailState.DATA0

class TH13(NCLGate):
    def evaluate(self, a, b, c):
        if any(x.is_null for x in (a,b,c)): return RailState.NULL
        return RailState.DATA1 if any(x.value for x in (a,b,c)) else RailState.DATA0

class TH23(NCLGate):
    def evaluate(self, a, b, c):
        if any(x.is_null for x in (a,b,c)): return RailState.NULL
        return RailState.DATA1 if sum(1 for x in (a,b,c) if x.value) >= 2 else RailState.DATA0

class TH33(NCLGate):
    def evaluate(self, a, b, c):
        if any(x.is_null for x in (a,b,c)): return RailState.NULL
        return RailState.DATA1 if all(x.value for x in (a,b,c)) else RailState.DATA0

class THmn(NCLGate):
    def __init__(self, m, n, name=None):
        super().__init__(name)
        self.m, self.n = m, n
    def evaluate(self, *inputs):
        if any(x.is_null for x in inputs): return RailState.NULL
        return RailState.DATA1 if sum(1 for x in inputs if x.value) >= self.m else RailState.DATA0

class NCLInverter(NCLGate):
    def evaluate(self, a):
        if a.is_null: return RailState.NULL
        return RailState.DATA0 if a.value else RailState.DATA1

class NCLRegister:
    def __init__(self, name, width=1):
        self.name  = name
        self.width = width
        self.rails = [DualRail(f"{name}[{i}]") for i in range(width)]
    def write(self, values):
        for r, v in zip(self.rails, values): r.set_data(v)
    def read(self):  return [r.value for r in self.rails]
    def reset(self):
        for r in self.rails: r.set_null()
    @property
    def is_null(self):     return all(r.is_null for r in self.rails)
    @property
    def is_complete(self): return all(r.is_data for r in self.rails)

class CompletionDetector:
    def __init__(self, outputs):
        self.outputs = outputs
    @property
    def complete(self): return all(o.is_data for o in self.outputs)
    @property
    def null(self):     return all(o.is_null for o in self.outputs)
