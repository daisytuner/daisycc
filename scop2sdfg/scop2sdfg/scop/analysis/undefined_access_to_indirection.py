from typing import Dict

from scop2sdfg.scop.computation.access import Access
from scop2sdfg.scop.computation.indirection import Indirection
from scop2sdfg.scop.symbols.constant import Constant
from scop2sdfg.scop.value import Value
from scop2sdfg.scop.undefined_value import UndefinedValue


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
