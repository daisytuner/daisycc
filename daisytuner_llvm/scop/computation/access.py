from __future__ import annotations

import dace
import islpy as isl

from typing import Dict, List, Set

from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
)

from daisytuner_llvm.scop.value import Value
from daisytuner_llvm.scop.undefined_value import UndefinedValue
from daisytuner_llvm.scop.symbols.constant import Constant
from daisytuner_llvm.scop.symbols.loop import Loop
from daisytuner_llvm.scop.symbols.memref import Memref


class Access(Value):

    _PLACEHOLDER_SYMBOLS = {f"o{i}": dace.symbol(f"o{i}") for i in range(10)}

    def __init__(
        self,
        reference: str,
        dtype: str,
        kind: str,
        instruction: str,
        incoming_value: str,
        array: str,
        expr: List[dace.symbolic.SymExpr],
    ) -> None:
        super().__init__(reference, dtype)

        self._kind = kind
        self._instruction = instruction
        self._array = array
        self._expr = expr

        self._arguments = set()
        if self._kind == "write":
            assert incoming_value

            if Value.is_llvm_value(incoming_value):
                self._arguments.add(UndefinedValue(incoming_value, dtype))
            else:
                self._arguments.add(
                    Constant(Constant.new_identifier(), dtype, incoming_value)
                )

    def __repr__(self) -> str:
        return self._reference

    def __str__(self) -> str:
        return self._reference

    def __hash__(self):
        return hash(self._reference)

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def array(self) -> str:
        return self._array

    def arguments(self) -> Set[Value]:
        return self._arguments

    def as_cpp(self) -> str:
        return "_in"

    def memlet(self) -> dace.Memlet:
        expr = self._array + "[" + ",".join([str(expr) for expr in self._expr]) + "]"
        return dace.Memlet(data=self._array, expr=expr)

    def validate(self) -> bool:
        for index in self._expr:
            if index.free_symbols.intersection(Access._PLACEHOLDER_SYMBOLS.values()):
                return False

        for arg in self._arguments:
            if isinstance(arg, UndefinedValue):
                return False

        return True

    def undefined_symbols(self) -> Set:
        undefined_symbols = set()
        for index in self._expr:
            undefined_symbols.update(
                index.free_symbols.intersection(Access._PLACEHOLDER_SYMBOLS.values())
            )

        return undefined_symbols

    @staticmethod
    def from_json(
        access: Dict,
        domain: isl.UnionSet,
        loops: List[Loop],
        memrefs: Dict[str, Memref],
    ) -> Access:
        mapping = isl.UnionMap.read_from_str(isl.DEFAULT_CONTEXT, access["relation"])
        mapping = mapping.intersect_domain(domain)
        mapping = mapping.gist_domain(domain)
        # mapping = mapping.project_out_all_params()

        memlet = mapping.to_str()
        memlet = memlet.replace("{", "").replace("}", "").strip()
        memlet = memlet.split("->")[-1]
        memlet = memlet.split(":")[0].strip()
        array, expr = memlet[: memlet.index("[")], memlet[memlet.index("[") + 1 : -1]
        if not expr:
            expr = "0"

        # Define symbols
        dimensions_to_iterators = {loop.dimension: loop.name for loop in loops.values()}
        dimensions = {
            dim: dace.symbolic.symbol(dim) for dim in dimensions_to_iterators.keys()
        }
        params = {f"p_{i}": dace.symbolic.symbol(f"p_{i}") for i in range(10)}
        known_symbols = {**Access._PLACEHOLDER_SYMBOLS, **dimensions, **params}

        # Convert to sympy
        indices = []
        for e in expr.split(","):
            e = e.split("=")[-1].strip()
            indices.append(e)
        indices = ",".join(indices)

        transformations = standard_transformations + (
            implicit_multiplication_application,
        )
        indices = parse_expr(
            indices,
            transformations=transformations,
            local_dict=known_symbols,
        )
        if not isinstance(indices, (list, tuple)):
            indices = [indices]

        # Represent as loops
        symbolic_indices = []
        for index in indices:
            symbolic_index = index.subs(dimensions_to_iterators)
            symbolic_index = symbolic_index.simplify()
            symbolic_indices.append(symbolic_index)

        # Metadata
        memref = memrefs[array]
        instruction = access["access_instruction"].strip()
        kind = access["kind"]
        if memref.kind == "value":
            if kind == "read":
                ref = memref.reference
            else:
                ref = Value.new_identifier()
        else:
            if kind == "read":
                ref = instruction.split("=")[0]
                ref = ref.strip()
            else:
                ref = Value.new_identifier()

        incoming_value = access["incoming_value"]
        if incoming_value:
            if "=" in incoming_value:
                incoming_value = incoming_value.split("=")[0].strip()
            else:
                incoming_value = incoming_value.split()[1]

        return Access(
            ref,
            memref.dtype,
            kind,
            instruction,
            incoming_value,
            array,
            symbolic_indices,
        )
