from typing import Set

from scop2sdfg.scop.value import Value


class UndefinedValue(Value):
    """
    The UndefinedValue is a placeholder for references.
    """

    def __init__(self, reference: str, dtype: str) -> None:
        super().__init__(reference, dtype)

    def __repr__(self) -> str:
        return self._reference

    def __str__(self) -> str:
        return self._reference

    def __hash__(self):
        return hash(self._reference)

    def validate(self) -> bool:
        return False

    def arguments(self) -> Set[Value]:
        raise NotImplementedError

    def as_cpp(self) -> str:
        raise NotImplementedError

    def as_sympy(self) -> str:
        raise NotImplementedError
