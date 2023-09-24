from typing import Set, Dict, List

from scop2sdfg.ir.value import Value
from scop2sdfg.ir.undefined_value import UndefinedValue
from scop2sdfg.ir.symbols.constant import Constant


class Phi(Value):
    def __init__(
        self, reference: str, dtype: str, condition: str, values: List[str]
    ) -> None:
        super().__init__(reference, dtype)

        self._condition = condition
        self._values = values

        self._arguments = set()
        if condition is not None:
            self._arguments.add(UndefinedValue(self._condition, dtype="i1"))

        for val in self._values:
            if Value.is_llvm_value(val):
                self._arguments.add(UndefinedValue(val, dtype=dtype))
            else:
                self._arguments.add(Constant(Constant.new_identifier(), dtype, val))

    def __repr__(self) -> str:
        return self._reference

    def __str__(self) -> str:
        return self._reference

    def __hash__(self):
        return hash(self._reference)

    def arguments(self) -> Set[Value]:
        return self._arguments

    def as_cpp(self) -> str:
        if self._condition is None:
            return f"{Value.canonicalize(self._values[0])}"
        else:
            return f"{Value.canonicalize(self._condition)} ? {Value.canonicalize(self._values[0])} : {Value.canonicalize(self._values[1])}"

    def validate(self) -> bool:
        for arg in self._arguments:
            if isinstance(arg, UndefinedValue):
                return False

        return True

    @property
    def code(self) -> str:
        return "phi"

    @property
    def name(self) -> str:
        return "phi"

    @staticmethod
    def from_string(instruction: str, branch_insts: Dict):
        ref, inst = instruction.split("=")
        ref = ref.strip()
        inst = inst.strip()

        _, dtype, tokens = inst.split(maxsplit=2)
        tokens = tokens.replace("[", "").replace("]", "").split(",")
        tokens = [token.strip() for token in tokens]
        incoming_edges = tokens[1::2]
        values = tokens[::2]
        if "undef" in values:
            return None

        if len(set(incoming_edges)) == 1 or len(set(values)) == 1:
            values = [values[0]]
            return Phi(ref, dtype, condition=None, values=values)

        for (cond, iftrue, iffalse) in branch_insts:
            branches = (iftrue, iffalse)
            if len(set(branches).intersection(incoming_edges)) == 2:
                vals = []
                for branch in branches:
                    vals.append(values[incoming_edges.index(branch)])

                return Phi(ref, dtype, cond, vals)

        last_edge = incoming_edges[-1]
        last_val = values[-1]
        for (cond, iftrue, iffalse) in branch_insts:
            if last_edge == iftrue:
                return Phi(ref, dtype, cond, [last_val, values[0]])
            elif last_edge == iffalse:
                return Phi(ref, dtype, cond, [values[0], last_val])

        return None
