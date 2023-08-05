from __future__ import annotations

import dace

from typing import Dict, List, Set

from daisytuner_llvm.scop.value import Value
from daisytuner_llvm.scop.undefined_value import UndefinedValue
from daisytuner_llvm.scop.symbols.constant import Constant


class Indirection(Value):
    def __init__(
        self,
        reference: str,
        dtype: str,
        kind: str,
        instruction: str,
        incoming_value: str,
        array: str,
        arguments: List[Value],
    ) -> None:
        super().__init__(reference, dtype)

        self._kind = kind
        self._instruction = instruction
        self._array = array
        self._expr = arguments

        self._arguments = set(arguments)
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

    @property
    def expr(self) -> List[Value]:
        return self._expr

    def arguments(self) -> Set[Value]:
        return self._arguments

    def as_cpp(self) -> str:
        return (
            "_in"
            + "["
            + ",".join(
                [
                    Value.canonicalize(str(expr))
                    for expr in self._expr
                    if not isinstance(expr, Constant)
                ]
            )
            + "]"
        )

    def memlet(self, data: dace.data.Data) -> dace.Memlet:
        expr = []
        for i, e in enumerate(self._expr):
            if isinstance(e, Constant):
                expr.append(e.as_cpp())
            else:
                expr.append(f"0:{data.shape[i]}")

        expr = self._array + "[" + ",".join(expr) + "]"
        return dace.Memlet(data=self._array, expr=expr)

    def validate(self) -> bool:
        for arg in self._arguments:
            if isinstance(arg, UndefinedValue):
                return False

        return True
