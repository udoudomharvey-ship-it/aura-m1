"""
aura_sdk.compiler
─────────────────
Language: Python
NCL expression compiler — runs offline, not in the hot path.

    from aura_sdk.compiler import compile, CompileResult
    cr = compile("AND(OR(a, b), NOT(c))")
    cr.print_truth_table()
"""
from .ncl_compiler import compile, CompileResult, ExpressionParser, NCLSynthesiser

__all__ = ["compile", "CompileResult", "ExpressionParser", "NCLSynthesiser"]
