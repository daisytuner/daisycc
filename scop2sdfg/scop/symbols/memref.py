from __future__ import annotations

import dace

from typing import List, Set, Dict

from scop2sdfg.scop.value import Value
from scop2sdfg.scop.symbols.constant import Constant


class Memref(Value):
    def __init__(
        self, reference: str, name: str, dtype: str, shape: List, kind: str
    ) -> None:
        super().__init__(reference=reference, dtype=dtype)
        self._name = name
        self._shape = shape
        self._kind = kind

    def __repr__(self) -> str:
        return self._reference

    def __str__(self) -> str:
        return self._reference

    def __hash__(self):
        return hash(self._reference)

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> List:
        return self._shape

    @property
    def kind(self) -> str:
        return self._kind

    def arguments(self) -> Set[Value]:
        return set()

    def as_cpp(self) -> str:
        return self._name

    def validate(self) -> bool:
        return True

    @staticmethod
    def from_json(desc: Dict) -> Memref:
        shape = []
        for dim in desc["sizes"]:
            if dim.isdigit():
                shape.append(int(dim))
            else:
                symbol = dace.symbol(name=Constant.new_identifier(), dtype=dace.int64)
                shape.append(symbol)

        reference = desc["variable"].strip()
        if "=" in reference:
            reference = reference.split()[0].strip()
        else:
            reference = reference.split()[-1].strip()

        return Memref(
            reference=reference,
            name=desc["name"],
            dtype=desc["type"],
            shape=shape,
            kind=desc["kind"],
        )
