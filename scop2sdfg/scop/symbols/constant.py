import struct
import ast

import random
import string

from typing import Set

from scop2sdfg.scop.value import Value


class Constant(Value):

    _IDENTIFIER = set()

    def __init__(self, reference: str, dtype: str, expression: str) -> None:
        assert Constant.is_constant(expression)
        super().__init__(reference, dtype)

        self._expression = expression

    def __repr__(self) -> str:
        return self._reference

    def __str__(self) -> str:
        return self._reference

    def __hash__(self):
        return hash(self._reference)

    def arguments(self) -> Set[Value]:
        return set()

    def as_cpp(self) -> str:
        if self._expression.startswith("0x"):
            return str(Constant.hextofloat(self._expression))
        else:
            return str(self._expression)

    def validate(self) -> bool:
        return True

    @staticmethod
    def is_constant(expression: str):
        if Value.is_llvm_value(expression):
            return False

        # Floating point as hex
        if expression.startswith("0x"):
            return True

        # C bool
        if expression in ["true", "false"]:
            return True

        #
        try:
            ast.literal_eval(expression)
            return True
        except:
            return False

    @staticmethod
    def hextofloat(hex: str) -> str:
        return str(struct.unpack("!d", bytes.fromhex(hex[2:]))[0])

    @staticmethod
    def new_identifier() -> str:
        new_ref = "".join(random.choice(string.ascii_uppercase) for _ in range(6))
        while new_ref in Constant._IDENTIFIER:
            new_ref = "".join(random.choice(string.ascii_uppercase) for _ in range(6))

        Constant._IDENTIFIER.add(new_ref)
        return new_ref
