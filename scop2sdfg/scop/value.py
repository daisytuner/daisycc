from __future__ import annotations

import dace
import random
import string

from typing import Set

from scop2sdfg.scop.types import TYPES


class Value:

    _IDENTIFIER = set()

    def __init__(self, reference: str, dtype: str) -> None:
        self._reference = reference
        if isinstance(dtype, dace.dtypes.typeclass):
            self._dtype = dtype
        else:
            self._dtype = TYPES[dtype]

    def __repr__(self) -> str:
        return self._reference

    def __str__(self) -> str:
        return self._reference

    def __hash__(self):
        return hash(self._reference)

    @property
    def reference(self) -> str:
        return self._reference

    @property
    def dtype(self):
        return self._dtype

    def arguments(self) -> Set[Value]:
        raise NotImplementedError

    def as_cpp(self) -> str:
        raise NotImplementedError

    def validate(self) -> bool:
        raise NotImplementedError

    @staticmethod
    def is_llvm_value(expr: str) -> bool:
        return expr.startswith("%") or expr.startswith("@")

    @staticmethod
    def canonicalize(expr: str) -> bool:
        return (
            expr.strip()
            .replace("%", "_")
            .replace("@", "_")
            .replace(".", "")
            .replace(",", "")
        )

    @staticmethod
    def new_identifier() -> str:
        new_ref = "".join(random.choice(string.ascii_uppercase) for _ in range(6))
        while new_ref in Value._IDENTIFIER:
            new_ref = "".join(random.choice(string.ascii_uppercase) for _ in range(6))

        Value._IDENTIFIER.add(new_ref)
        return "%" + new_ref
