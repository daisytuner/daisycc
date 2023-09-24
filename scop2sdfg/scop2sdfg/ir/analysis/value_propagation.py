from scop2sdfg.ir.symbols.constant import Constant


def value_propagation(scop):
    """
    Replaces UndefinedValue arguments by specific type in computations of scop.
    """
    for _, comp in scop._computations.items():
        new_args = set()
        for arg in comp.arguments():
            propagated = False
            if isinstance(arg, Constant):
                new_args.add(arg)
                propagated = True
            elif arg.reference in scop._parameters:
                new_args.add(scop._parameters[arg.reference])
                propagated = True
            elif arg.reference in scop._computations:
                new_args.add(scop._computations[arg.reference])
                propagated = True
            else:
                for _, mems in scop._memory_accesses.items():
                    if arg.reference in mems and mems[arg.reference].kind == "read":
                        new_args.add(mems[arg.reference])
                        propagated = True
                        break

                for _, ls in scop._loops.items():
                    if arg.reference in ls:
                        new_args.add(ls[arg.reference])
                        propagated = True
                        break

            # Speculative: Irrelevant instruction
            # if not propagated:
            #     new_args.add(arg)

        comp._arguments = new_args

    for statement in scop._memory_accesses:
        for _, access in scop._memory_accesses[statement].items():
            new_args = set()
            for arg in access.arguments():
                propagated = False
                if isinstance(arg, Constant):
                    new_args.add(arg)
                    propagated = True
                elif arg.reference in scop._parameters:
                    new_args.add(scop._parameters[arg.reference])
                    propagated = True
                elif arg.reference in scop._computations:
                    new_args.add(scop._computations[arg.reference])
                    propagated = True
                else:
                    for _, mems in scop._memory_accesses.items():
                        if arg.reference in mems and mems[arg.reference].kind == "read":
                            new_args.add(mems[arg.reference])
                            propagated = True
                            break

                    for _, ls in scop._loops.items():
                        if arg.reference in ls:
                            new_args.add(ls[arg.reference])
                            propagated = True
                            break

                # Speculative: Irrelevant instruction
                # if not propagated:
                #     new_args.add(arg)

            access._arguments = new_args
