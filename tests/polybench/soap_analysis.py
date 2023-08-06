# Copyright 2023 Lukas Truemper. All rights reserved.
import dace
import json

from pathlib import Path

from dace.symbolic import pystr_to_symbolic
from daisytuner.profiling.experimental.soap import Solver, perform_soap_analysis

if __name__ == "__main__":
    path = Path(__file__).parent
    bench = "trmm"

    solver = Solver(address="localhost", port=30000)
    solver.connect()

    # "I" is reserved for complex numbers
    sdfg = dace.SDFG.from_file(path / bench / f"{bench}.sdfg")
    sdfg.replace("I", "__I")

    cache_size = 30 * 1e6
    num_processors = 12

    bytes_per_element = 0
    for _, desc in sdfg.arrays.items():
        b = desc.dtype.bytes
        if b > bytes_per_element:
            bytes_per_element = b
    cache_size_elements = int(cache_size / bytes_per_element)

    result = perform_soap_analysis(
        sdfg=sdfg,
        solver=solver,
    )
    Q = pystr_to_symbolic(str(result.Q))

    # SOAP messes with the symbols in the SDFG, e.g., changes the case
    symbol_map = {
        dace.symbol("Ss"): cache_size_elements,
        dace.symbol("p"): num_processors,
    }
    for sym in Q.free_symbols:
        if str(sym) in sdfg.constants:
            symbol_map[sym] = sdfg.constants[str(sym)]
            continue

        s = str(sym).upper()
        if s in sdfg.constants:
            symbol_map[sym] = sdfg.constants[s]

    Q_eval = float(dace.symbolic.evaluate(Q, symbols=symbol_map))
    res = {
        "Q": str(Q),
        "Q_eval": Q_eval,
        "P": 12,
        "Ss": cache_size_elements,
        "element_size": bytes_per_element,
    }
    with open(path / bench / "soap.json", "w") as handle:
        json.dump(res, handle, indent=4)
