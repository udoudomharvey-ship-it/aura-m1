"""aura_sdk.ncl.simulator — Event-driven NCL circuit simulator."""
from __future__ import annotations
from dataclasses import dataclass, field
import time
from .gates import DualRail, RailState, NCLGate


@dataclass
class SimEvent:
    time_ns: float
    gate_name: str
    input_states: list
    output_state: RailState
    energy_pJ: float = 0.0


@dataclass
class CircuitNode:
    gate: NCLGate
    inputs: list
    output: DualRail
    energy_pJ: float = 0.05


@dataclass
class SimStats:
    total_energy_pJ: float = 0.0
    gate_firings:    int   = 0
    sim_time_us:     float = 0.0


@dataclass
class SimResult:
    output_values:   dict
    node_outputs:    dict
    events:          list
    total_energy_pJ: float
    latency_ns:      float
    gate_firings:    int

    @property
    def idle_power_W(self): return 0.0

    def summary(self):
        return (f"── NCL Simulation ─────────────────────\n"
                f"  Outputs    : {self.output_values}\n"
                f"  Firings    : {self.gate_firings}\n"
                f"  Latency    : {self.latency_ns:.1f} ns\n"
                f"  Energy     : {self.total_energy_pJ:.3f} pJ\n"
                f"  Idle power : 0.00 W  ← SIMULATED ZERO (NCL NULL — no hardware measurement)\n"
                f"───────────────────────────────────────")

    def __repr__(self): return self.summary()


class NCLSimulator:
    GATE_ENERGY_pJ = 0.05
    PROP_DELAY_ns  = 0.5

    def __init__(self):
        self._inputs:  dict = {}
        self._outputs: dict = {}
        self._nodes:   list = []
        self._events:  list = []
        self._t_ns = 0.0
        self.stats = SimStats()

    def input(self, name):
        r = DualRail(name)
        self._inputs[name] = r
        return r

    def output(self, name, rail):
        self._outputs[name] = rail
        return rail

    def add_gate(self, gate, inputs, out_name=None, energy_pJ=None):
        out_name = out_name or f"{gate.name}_out"
        out = DualRail(out_name)
        self._nodes.append(CircuitNode(gate, inputs, out, energy_pJ or self.GATE_ENERGY_pJ))
        return out

    def inject(self, values):
        for name, v in values.items():
            self._inputs[name].set_data(v)

    def reset(self):
        for r in self._inputs.values(): r.set_null()
        for n in self._nodes:           n.output.set_null()
        self._t_ns = 0.0

    def run(self, verbose=False):
        self._events.clear()
        changed = True
        while changed:
            changed = False
            for node in self._nodes:
                prev = node.output.state
                new  = node.gate(*node.inputs)
                if new != prev:
                    changed = True
                    node.output.set(new)
                    self._t_ns += self.PROP_DELAY_ns
                    if new != RailState.NULL:
                        ev = SimEvent(self._t_ns, node.gate.name,
                                      [i.state for i in node.inputs], new, node.energy_pJ)
                        self._events.append(ev)
                        self.stats.total_energy_pJ += node.energy_pJ
                        self.stats.gate_firings    += 1
                        if verbose: print(f"  {ev}")

        return SimResult(
            output_values={n: r.value for n, r in self._outputs.items()},
            node_outputs ={n.output.name: n.output.value for n in self._nodes},
            events        =self._events.copy(),
            total_energy_pJ=self.stats.total_energy_pJ,
            latency_ns    =self._t_ns,
            gate_firings  =self.stats.gate_firings,
        )
