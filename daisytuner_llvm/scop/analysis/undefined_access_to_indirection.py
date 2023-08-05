from typing import Dict

from daisytuner_llvm.scop.computation.access import Access
from daisytuner_llvm.scop.computation.indirection import Indirection
from daisytuner_llvm.scop.symbols.constant import Constant
from daisytuner_llvm.scop.value import Value
from daisytuner_llvm.scop.undefined_value import UndefinedValue


def undefined_access_to_indirection(scop, geps: Dict) -> Dict:
    for stmt_name in scop._memory_accesses:
        indirections = {}
        for ref, access in scop._memory_accesses[stmt_name].items():
            if len(access.undefined_symbols()) == 0:
                continue

            instruction = access.instruction
            if "phi" in instruction:
                continue

            # Jump to getelementptr
            gep = instruction.split(",")[1].strip()
            gep = gep.split()[-1]
            gep = geps[gep]

            # Extract indices
            indices = gep.split(",")
            indices = list(map(lambda token: token.strip(), indices))[2:]
            indices = [ind.split() for ind in indices]

            # GEP to symbolic expression
            arguments = []
            for (dtype, expr) in indices[1:]:
                if Value.is_llvm_value(expr):
                    symbolic_index = UndefinedValue(expr, dtype)
                else:
                    symbolic_index = Constant(Constant.new_identifier(), dtype, expr)

                arguments.append(symbolic_index)

            indirection = Indirection(
                ref,
                access.dtype,
                access.kind,
                access.instruction,
                "",
                access.array,
                arguments,
            )
            indirections[ref] = indirection

        for ref, ind in indirections.items():
            scop._memory_accesses[stmt_name][ref] = ind
