import dace

from typing import Set

from daisytuner_llvm.scop.value import Value


class Parameter(Value):
    def __init__(self, reference: str, dtype: str, name: str) -> None:
        super().__init__(reference=reference, dtype=dtype)
        assert self._dtype in (dace.int8, dace.int16, dace.int32, dace.int64)

        self._name = name

    def __repr__(self) -> str:
        return self._reference

    def __str__(self) -> str:
        return self._reference

    def __hash__(self):
        return hash(self._reference)

    @property
    def name(self) -> str:
        return self._name

    def arguments(self) -> Set[Value]:
        return set()

    def as_cpp(self) -> str:
        return self._name

    def validate(self) -> bool:
        return True
