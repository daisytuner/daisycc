from __future__ import annotations

import dace
import islpy as isl

from typing import Dict
from collections import OrderedDict

from scop2sdfg.scop.ast import ASTBuilder

from scop2sdfg.scop.computation.computation import Computation
from scop2sdfg.scop.computation.access import Access

from scop2sdfg.scop.symbols.parameter import Parameter
from scop2sdfg.scop.symbols.memref import Memref
from scop2sdfg.scop.symbols.loop import Loop

from scop2sdfg.scop.analysis import (
    value_propagation,
    undefined_access_to_indirection,
)


class Scop:
    """
    The Scop class represents scops extracted from LLVM IR (control-centric) in a two-level representation suitable for the conversion to an SDFG (control- and data-centric).
    """

    def __init__(self, name: str, source: str) -> None:
        self._name = name
        self._source = source

        self._shape_inference = None

        ## Level I: control-centric (AST)
        # Scop statement -> DaCe state
        self._context = None
        self._statements = {}
        self._dependencies = None
        self._schedule = None
        self._ast = None

        ## Level II: data-centric (data, computation and symbols)

        # Symbols of the SDFG
        # a. Arguments: External input to the Scop / SDFG
        #   I.e., arrays, phis, scalar arguments, parameters
        self._memrefs = OrderedDict()
        self._parameters = OrderedDict()
        # b. Loops: Iteration variables
        self._loops = {}

        # Instructions
        # Scop: Everything is an instruction in LLVM
        # SDFG: Distinguishes memory accesses (memlet), computation (tasklet)
        self._computations = {}
        self._memory_accesses = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> str:
        return self._source

    @property
    def ast(self) -> isl.AstBuild:
        return self._ast

    def arguments(self) -> Dict:
        arguments = OrderedDict()
        for ref, item in self._memrefs.items():
            arguments[ref] = item
        for ref, item in self._scalars.items():
            arguments[ref] = item
        for ref, item in self._parameters.items():
            arguments[ref] = item
        return arguments

    def validate(self) -> bool:
        for ref, value in self._memrefs.items():
            assert value.validate()
        for ref, value in self._parameters.items():
            assert value.validate()
        for ref, value in self._computations.items():
            assert value.validate()
        for stmt in self._memory_accesses:
            for ref, value in self._memory_accesses[stmt].items():
                assert value.validate()

        return True

    @staticmethod
    def from_json(source: str, desc: Dict) -> Scop:
        scop = Scop(desc["name"], source)

        ## Level I: control-centric (AST)
        scop._context = isl.Set.read_from_str(isl.DEFAULT_CONTEXT, desc["context"])

        ctx = isl.DEFAULT_CONTEXT
        raw = isl.UnionMap.read_from_str(ctx, desc["dependencies"]["RAW"])
        war = isl.UnionMap.read_from_str(ctx, desc["dependencies"]["WAR"])
        waw = isl.UnionMap.read_from_str(ctx, desc["dependencies"]["WAW"])
        scop._dependencies = raw.union(war).union(waw)

        domains = None
        for statement in desc["statements"]:
            name = statement["name"]
            scop._statements[name] = {
                "domain": isl.UnionSet.read_from_str(ctx, statement["domain"]),
                "iterators": None,
                "dimensions": None,
                "loops": statement["loops"],
            }
            if domains is None:
                domains = scop._statements[name]["domain"]
            else:
                domains = domains.union(scop._statements[name]["domain"])

        scop._schedule = isl.UnionMap.read_from_str(ctx, desc["schedule"])
        scop._schedule = scop._schedule.intersect_domain(domains)

        builder = ASTBuilder()
        scop._ast = builder.create(
            scop._statements,
            scop._context,
            scop._schedule,
            scop._dependencies,
        )

        ## Level II: data-centric (data, computation and symbols)

        ## Arguments

        scop._shape_inference = desc["access_range"]

        # Memrefs (arrays)
        for array in desc["arrays"]:
            if array["kind"] != "array":
                continue
            memref = Memref.from_json(array)
            scop._memrefs[memref.name] = memref

        # Scalars: phis and values
        for array in desc["arrays"]:
            if array["kind"] == "array":
                continue
            memref = Memref.from_json(array)
            scop._memrefs[memref.name] = memref

        # Parameters
        for param in desc["parameters"]:
            name = param["name"].split("@")[0]
            dtype = param["type"]

            if "=" in param["variable"]:
                reference = param["variable"].split("=")[0]
            else:
                reference = param["variable"].split()[-1]

            scop._parameters[reference] = Parameter(reference, dtype, name)

        # Loops
        for name, statement in scop._statements.items():
            domain = statement["domain"]
            if domain.is_empty():
                continue

            k = 0
            set_ = domain.as_set()
            scop._loops[name] = {}
            for i in range(set_.n_dim()):
                dim_name = set_.get_dim_name(isl.dim_type.out, i)
                if not dim_name or dim_name is None:
                    continue

                loop = statement["loops"][i]
                reference, phi = loop["induction_variable"].split("=")
                reference = reference.strip()
                dtype = phi.split()[1].strip()

                iterator = statement["iterators"][k]
                scop._loops[name][reference] = Loop(
                    reference=reference,
                    dtype=dtype,
                    name=iterator,
                    dimension=dim_name,
                )

                k = k + 1

            assert len(scop._loops[name]) == len(statement["iterators"])

        ## Instructions

        # Memory accesses
        for statement in desc["statements"]:
            domain = isl.UnionSet.read_from_str(ctx, statement["domain"])
            if domain.is_empty():
                continue

            stmt_name = statement["name"]
            loops = scop._loops[stmt_name]
            scop._memory_accesses[stmt_name] = {}
            for access_desc in statement["accesses"]:
                access = Access.from_json(
                    access_desc, domain=domain, loops=loops, memrefs=scop._memrefs
                )

                assert access.reference not in scop._memory_accesses[stmt_name]
                scop._memory_accesses[stmt_name][access.reference] = access

        # Computations
        getelementptrs = {}
        for instruction in desc["instructions"].replace("\\n", "\n").splitlines():
            instruction = instruction.strip()
            if "getelementptr" in instruction:
                ref, inst = instruction.split("=")
                ref = ref.strip()
                inst = inst.strip()
                getelementptrs[ref] = inst
                continue

            try:
                computation = Computation.from_str(instruction)
                scop._computations[computation.reference] = computation
            except Computation.InvalidComputation:
                continue

        value_propagation(scop)
        undefined_access_to_indirection(scop, getelementptrs)
        value_propagation(scop)

        return scop
