from scop2sdfg.scop.undefined_value import UndefinedValue
from scop2sdfg.scop.symbols.constant import Constant


def value_propagation(scop):
    """
    Replaces UndefinedValue arguments by specific type in computations of scop.
    """
    for _, comp in scop._computations.items():
        new_args = set()
        for arg in comp.arguments():
            if isinstance(arg, Constant):
                new_args.add(arg)
                continue

            ref = arg.reference
            if ref in scop._parameters:
                new_args.add(scop._parameters[ref])
            elif ref in scop._computations:
                new_args.add(scop._computations[ref])
            else:
                for _, mems in scop._memory_accesses.items():
                    if ref in mems and mems[ref].kind == "read":
                        new_args.add(mems[ref])
                        break

                for _, ls in scop._loops.items():
                    if ref in ls:
                        new_args.add(ls[ref])
                        break

        comp._arguments = new_args

    for statement in scop._memory_accesses:
        for _, access in scop._memory_accesses[statement].items():
            new_args = set()
            for arg in access.arguments():
                if isinstance(arg, Constant):
                    new_args.add(arg)
                    continue

                ref = arg.reference
                if ref in scop._parameters:
                    new_args.add(scop._parameters[ref])
                elif ref in scop._computations:
                    new_args.add(scop._computations[ref])
                else:
                    for _, mems in scop._memory_accesses.items():
                        if ref in mems and mems[ref].kind == "read":
                            new_args.add(mems[ref])
                            break

                    for _, ls in scop._loops.items():
                        if ref in ls:
                            new_args.add(ls[ref])
                            break

            access._arguments = new_args
