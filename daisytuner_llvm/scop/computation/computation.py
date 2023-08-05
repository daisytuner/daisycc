from __future__ import annotations

from typing import Set

from daisytuner_llvm.scop.value import Value
from daisytuner_llvm.scop.undefined_value import UndefinedValue
from daisytuner_llvm.scop.symbols.constant import Constant


class Computation(Value):

    CONSTANTS = set()

    class InvalidComputation(Exception):
        def __init__(self, *args: object) -> None:
            super().__init__(*args)

    # Mapping LLVM comparators to comparators
    _COMPARATORS = {
        "eq": "==",
        "ne": "!=",
        "ugt": ">",
        "uge": ">=",
        "ult": "<",
        "ule": "<=",
        "sgt": ">",
        "sge": ">=",
        "slt": "<",
        "ogt": ">",
        "olt": "<",
        "ole": "<=",
    }

    # Mapping token positions to code
    _INSTRUCTIONS = {
        # Unary
        "fneg": (-2, (-1,), "-1 * $1", "-1 * $1"),
        # BINARY
        "add": (-3, (-2, -1), "$1 + $2", "$1 + $2"),
        "fadd": (-3, (-2, -1), "$1 + $2", "$1 - $2"),
        "sub": (-3, (-2, -1), "$1 - $2", "$1 - $2"),
        "fsub": (-3, (-2, -1), "$1 - $2", "$1 - $2"),
        "mul": (-3, (-2, -1), "$1 * $2", "$1 * $2"),
        "fmul": (-3, (-2, -1), "$1 * $2", "$1 * $2"),
        "udiv": (-3, (-2, -1), "$1 / $2", "$1 / $2"),
        "sdiv": (-3, (-2, -1), "$1 / $2", "$1 / $2"),
        "fdiv": (-3, (-2, -1), "$1 / $2", "$1 / $2"),
        "srem": (-3, (-2, -1), "$1 % $2", "$1 % $2"),
        "frem": (-3, (-2, -1), "$1 % $2", "$1 % $2"),
        "urem": (-3, (-2, -1), "($0) $1 % ($0) $2", "$1 % $2"),
        # BITWISE BINARY
        "shl": (-3, (-2, -1), "$1 << $2", "$1 << $2"),
        "lshr": (-3, (-2, -1), "(unsigned $0) $1 >> $2", ""),
        "ashr": (
            -3,
            (-2, -1),
            "($1 < 0 && $ > 0) ? $1 >> $2 | (~0U >> $2) : $1 >> $2",
            "",
        ),
        "and": (-3, (-2, -1), "$1 & $2", "$1 & $2"),
        "or": (-3, (-2, -1), "$1 | $2", "$1 | $2"),
        "xor": (-3, (-2, -1), "$1 ^ $2", "$1 ^ $2"),
        # CONVERSION
        "trunc": (-1, (-3,), "($0) $1", ""),
        "sext": (-1, (-3,), "($0) $1", ""),
        "zext": (-1, (-3,), "($0) $1", ""),
        "fptrunc": (-1, (-3,), "($0) $1", ""),
        "fpext": (-1, (-3,), "($0) $1", ""),
        "fptosi": (-1, (-3,), "($0) $1", ""),
        "sitofp": (-1, (-3,), "($0) $1", ""),
        # OTHER
        "icmp": (-3, (-2, -1, -4), "$1 $3 $2", "$1 $3 $2"),
        "fcmp": (-3, (-2, -1, -4), "$1 $3 $2", "$1 $3 $2"),
        "select": (-2, (-5, -3, -1), "$1 ? $2 : $3", ""),
    }

    # Mapping argument positions to code
    _FUNCTIONS = {
        # C/C++ Standard Instrinsics
        "abs": ("abs($1)", ""),
        "smax": ("max($1, $2)", ""),
        "smin": ("min($1, $2)", ""),
        "minnum": ("min($1, $2)", ""),
        "maxnum": ("max($1, $2)", ""),
        "sqrt": ("sqrt($1)", ""),
        "powi": ("pow($1, $2)", ""),
        "sin": ("sin($1)", ""),
        "cos": ("cos($1)", ""),
        "pow": ("pow($1, $2)", ""),
        "exp": ("exp($1)", ""),
        "exp2": ("exp2($1)", ""),
        "log": ("log($1)", ""),
        "log10": ("log10($1)", ""),
        "log2": ("log2($1)", ""),
        "fma": ("$3 + ($1 * $2)", ""),
        "fabs": ("abs($1)", ""),
        "copysign": ("copysign($1, $2)", ""),
        "floor": ("floor($1)", ""),
        "ceil": ("ceil($1)", ""),
        "trunc": ("trunc($1)", ""),
        "rint": ("rint($1)", ""),
        "nearbyint": ("nearbyint($1)", ""),
        "round": ("round($1)", ""),
        "roundeven": ("roundeven($1)", ""),
        "lround": ("lround($1)", ""),
        "llround": ("llround($1)", ""),
        "lrint": ("lrint($1)", ""),
        "llrint": ("llrint($1)", ""),
        # Specialized Arithmetic Instrinsics
        "fmuladd": ("$3 + ($1 * $2)", ""),
    }

    def __init__(self, reference: str, code: str) -> None:
        self._code = code
        self._name = None
        self._cpp_code = None
        self._sympy_code = None
        self._arguments = set()

        # Analyse instructions and set properties
        inst = code
        tokens = [token.strip().replace(",", "") for token in inst.split()]
        if "call" in tokens:
            # LLVM function
            start_token = None
            for t in range(len(tokens)):
                if tokens[t].startswith("@"):
                    start_token = t
                    self._name = tokens[t].split(".")[1]
                    break

            super().__init__(reference=reference, dtype=tokens[start_token - 1])

            self._cpp_code, self._sympy_code = Computation._FUNCTIONS[self._name]

            function = " ".join(tokens[start_token:])
            arguments = function[function.find("(") + 1 : function.find(")")].split()

            self._cpp_code = self._cpp_code.replace("$0", self._dtype.ctype)
            self._sympy_code = self._sympy_code.replace("$0", self._dtype.ctype)
            for i in range(0, len(arguments), 2):
                arg, dtype = arguments[i + 1], arguments[i]
                if Value.is_llvm_value(arg):
                    arg = UndefinedValue(arg, dtype)
                else:
                    arg = Constant(Constant.new_identifier(), dtype, arg)

                self._cpp_code = self._cpp_code.replace(
                    f"${int(i/2) + 1}", Value.canonicalize(str(arg))
                )
                self._sympy_code = self._sympy_code.replace(
                    f"${int(i/2) + 1}", Value.canonicalize(str(arg))
                )
                self._arguments.add(arg)
        else:
            # LLVM instruction
            self._name = tokens[0]
            (
                ind_dtype,
                ind_arguments,
                cpp_code,
                sympy_code,
            ) = Computation._INSTRUCTIONS[self._name]
            dtype = tokens[ind_dtype]
            arguments = [tokens[i] for i in ind_arguments]

            super().__init__(reference=reference, dtype=dtype)

            self._cpp_code = cpp_code.replace("$0", self._dtype.ctype)
            self._sympy_code = sympy_code.replace("$0", self._dtype.ctype)
            for i in range(len(arguments)):
                arg = arguments[i]
                if arg in Computation._COMPARATORS:
                    arg = Computation._COMPARATORS[arg]
                    self._cpp_code = self._cpp_code.replace(f"${i+1}", arg)
                    self._sympy_code = self._sympy_code.replace(f"${i+1}", arg)
                else:
                    if Value.is_llvm_value(arg):
                        arg = UndefinedValue(arg, dtype)
                    else:
                        # TODO: arg may have different dtype
                        arg = Constant(Constant.new_identifier(), dtype, arg)

                    self._cpp_code = self._cpp_code.replace(
                        f"${i+1}", Value.canonicalize(str(arg))
                    )
                    self._sympy_code = self._sympy_code.replace(
                        f"${i+1}", Value.canonicalize(str(arg))
                    )
                    self._arguments.add(arg)

    def __repr__(self) -> str:
        return self._reference

    def __str__(self) -> str:
        return self._reference

    def __hash__(self):
        return hash(self._reference)

    def arguments(self) -> Set[Value]:
        return self._arguments

    def as_cpp(self) -> str:
        return self._cpp_code

    def validate(self) -> bool:
        for arg in self._arguments:
            if isinstance(arg, UndefinedValue):
                return False

        return True

    @property
    def code(self) -> str:
        return self._code

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def from_str(instruction: str) -> Computation:
        try:
            reference, instruction = instruction.strip().split("=")
            # The following should all be handled by Scop separately
            assert "load" not in instruction
            assert "store" not in instruction
            assert "phi" not in instruction
            assert "getelementptr" not in instruction
        except:
            raise Computation.InvalidComputation

        reference = reference.strip()
        instruction = instruction.strip()
        return Computation(reference=reference, code=instruction)
