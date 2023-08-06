import dace

from typing import Set

from scop2sdfg.scop.value import Value


class Loop(Value):
    def __init__(self, reference: str, dtype: str, name: str, dimension: str) -> None:
        super().__init__(reference, dtype)
        assert self._dtype in (dace.int8, dace.int16, dace.int32, dace.int64)

        self._name = name
        self._dimension = dimension
        self._shift = None

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
    def dimension(self) -> str:
        return self._dimension

    def arguments(self) -> Set[Value]:
        return set()

    def as_cpp(self) -> str:
        return self._name

    def validate(self) -> bool:
        return True
