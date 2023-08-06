import copy
import dace
import sympy
import islpy as isl

from pathlib import Path
from typing import Dict

from dace.sdfg.utils import consolidate_edges
from dace.frontend.python.astutils import negate_expr

from daisytuner.optimization.canonicalization import (
    AutoParallelization,
    LoopSimplification,
    GreedyLoopFusion,
)

from scop2sdfg.scop.scop import Scop
from scop2sdfg.scop.value import Value
from scop2sdfg.scop.computation.access import Access
from scop2sdfg.scop.computation.indirection import Indirection

from scop2sdfg.codegen.analysis import infer_shape
from scop2sdfg.codegen.isl import (
    to_sympy,
    sympy_to_pystr,
    extract_end_cond,
)


class Generator:
    def __init__(self, sdfg: dace.SDFG, scop: Scop) -> None:
        self._sdfg = sdfg
        self._scop = scop

        self._inputs = set()
        self._outputs = set()
        self._surrounding_loops = []

    def _visit(self, ast_node: isl.AstNode, loop_ranges, constraints):
        if ast_node.get_type() == isl.ast_node_type.block:
            first, last = self._visit_block(ast_node, loop_ranges, constraints)
        elif ast_node.get_type() == isl.ast_node_type.for_:
            first, last = self._visit_for(ast_node, loop_ranges, constraints)
        elif ast_node.get_type() == isl.ast_node_type.if_:
            first, last = self._visit_if(ast_node, loop_ranges, constraints)
        elif ast_node.get_type() == isl.ast_node_type.user:
            first, last = self._visit_user(ast_node, loop_ranges, constraints)
        else:
            raise NotImplementedError
        return first, last

    def _visit_block(self, ast_node: isl.AstNode, loop_ranges, constraints):
        node_list = ast_node.block_get_children()
        n_children = node_list.n_ast_node()

        states = []
        for child_node in [node_list.get_at(i) for i in range(n_children)]:
            ret_val = self._visit(child_node, loop_ranges.copy(), constraints)
            s1, s2 = ret_val
            states.append((s1, s2))

        if states:
            for (_, s1), (s2, _) in zip(states[:-1], states[1:]):
                self._sdfg.add_edge(s1, s2, dace.sdfg.InterstateEdge())
            return states[0][0], states[-1][1]
        else:
            empty_state = self._sdfg.add_state(f"block_{len(self._sdfg.nodes())}")
            return empty_state, empty_state

    def _visit_if(self, ast_node: isl.AstNode, loop_ranges, constraints):
        # Add a guard state
        if_guard = self._sdfg.add_state("if_guard")
        end_if_state = self._sdfg.add_state("end_if")

        # Generate conditions
        if_cond_sym = to_sympy(ast_node.if_get_cond())
        if_cond_str = sympy_to_pystr(if_cond_sym)
        else_cond_sym = negate_expr(if_cond_sym)
        else_cond_str = sympy_to_pystr(else_cond_sym)

        then_node = ast_node.if_get_then_node()
        if_constraints = constraints.copy()
        if_constraints.append(if_cond_sym)
        first_if_state, last_if_state = self._visit(
            then_node, loop_ranges.copy(), if_constraints
        )

        # Connect the states
        self._sdfg.add_edge(
            if_guard, first_if_state, dace.sdfg.InterstateEdge(if_cond_str)
        )
        self._sdfg.add_edge(last_if_state, end_if_state, dace.sdfg.InterstateEdge())

        if ast_node.if_has_else_node():
            else_node = ast_node.if_get_else_node()
            else_constraints = constraints.copy()
            else_constraints.append(else_cond_sym)
            first_else_state, last_else_state = self._visit(
                else_node, loop_ranges.copy(), else_constraints
            )

            # Connect the states
            self._sdfg.add_edge(
                if_guard, first_else_state, dace.sdfg.InterstateEdge(else_cond_str)
            )
            self._sdfg.add_edge(
                last_else_state, end_if_state, dace.sdfg.InterstateEdge()
            )
        else:
            self._sdfg.add_edge(
                if_guard, end_if_state, dace.sdfg.InterstateEdge(else_cond_str)
            )
        return if_guard, end_if_state

    def _visit_for(self, ast_node: isl.AstNode, loop_ranges, constraints):
        iter_sympy = to_sympy(ast_node.for_get_iterator())
        iterator_var = sympy_to_pystr(iter_sympy)
        self._surrounding_loops.append(iterator_var)

        init_sympy = to_sympy(ast_node.for_get_init())
        init_str = sympy_to_pystr(init_sympy)

        cond_sympy = to_sympy(ast_node.for_get_cond())
        end_sympy = extract_end_cond(cond_sympy, iter_sympy)
        condition_str = sympy_to_pystr(cond_sympy)

        step_sym = to_sympy(ast_node.for_get_inc())
        incr_str = sympy_to_pystr(sympy.Add(iter_sympy, step_sym))

        loop_rng = dace.subsets.Range([(init_sympy, end_sympy, step_sym)])
        loop_ranges.append((iterator_var, loop_rng))

        is_parallel = ast_node.get_annotation().user.is_parallel
        if is_parallel:
            state = self._sdfg.add_state(f"MapState_{len(self._sdfg.nodes())}")
            subset = loop_rng

            map_nodes = dace.nodes.Map(
                label="map", params=[iterator_var], ndrange=subset
            )

            entry = dace.nodes.MapEntry(map_nodes)
            exit = dace.nodes.MapExit(map_nodes)
            state.add_nodes_from([entry, exit])

            # create a new SDFG for the map body
            body_sdfg = dace.SDFG("{}_body".format(entry.label))

            # add all arrays of SDFG to the body-SDFG
            for arr_label, arr in self._sdfg.arrays.items():
                arr_copy = copy.deepcopy(arr)
                arr_copy.transient = False
                body_sdfg.add_datadesc(arr_label, arr_copy)

            body_sdfg.symbols.update(self._sdfg.symbols)

            # walk and add the states to the body_sdfg
            pv = Generator(sdfg=body_sdfg, scop=self._scop)
            pv._surrounding_loops = self._surrounding_loops
            (
                _,
                _,
            ) = pv._visit(ast_node.for_get_body(), loop_ranges.copy(), constraints)
            body = state.add_nested_sdfg(body_sdfg, self._sdfg, pv._inputs, pv._outputs)

            for array in self._sdfg.arrays:
                if array not in pv._inputs and array not in pv._outputs:
                    del body_sdfg.arrays[array]

            for arr_name in pv._inputs:
                read_node = state.add_read(arr_name)
                arr = body_sdfg.arrays[arr_name]
                subset = dace.subsets.Range.from_array(arr)
                memlet = dace.Memlet(data=arr_name, subset=subset)

                state.add_memlet_path(
                    read_node,
                    entry,
                    body,
                    memlet=memlet,
                    dst_conn=arr_name,
                    propagate=False,
                )
            if len(body.in_connectors) == 0:
                state.add_edge(entry, None, body, None, dace.Memlet())

            for arr_name in pv._outputs:
                write_node = state.add_write(arr_name)
                arr = body_sdfg.arrays[arr_name]
                subset = dace.subsets.Range.from_array(arr)
                memlet = dace.Memlet(data=arr_name, subset=subset)

                state.add_memlet_path(
                    body,
                    exit,
                    write_node,
                    memlet=memlet,
                    src_conn=arr_name,
                    dst_conn=None,
                    propagate=False,
                )
            if len(body.out_connectors) == 0:
                state.add_edge(body, None, exit, None, dace.Memlet())

            self._inputs.update(pv._inputs)
            self._outputs.update(pv._outputs)

            self._surrounding_loops.pop(-1)
            return state, state
        else:
            body_begin, body_end = self._visit(
                ast_node.for_get_body(), loop_ranges.copy(), constraints
            )

            if iterator_var not in self._sdfg.symbols:
                self._sdfg.add_symbol(iterator_var, dace.int64)

            if body_begin == body_end:
                body_end = None

            loop_result = self._sdfg.add_loop(
                before_state=None,
                loop_state=body_begin,
                loop_end_state=body_end,
                after_state=None,
                loop_var=iterator_var,
                initialize_expr=init_str,
                condition_expr=condition_str,
                increment_expr=incr_str,
            )
            before_state, guard, after_state = loop_result

            self._surrounding_loops.pop(-1)
            return before_state, after_state

    def _visit_user(self, ast_node: isl.AstNode, loop_ranges, constraints):
        ast_expr = ast_node.user_get_expr()
        if ast_expr.get_op_type() != isl.ast_expr_op_type.call:
            raise NotImplementedError

        # Define state
        stmt_name = ast_expr.get_op_arg(0).to_C_str()
        state = self._sdfg.add_state("state_" + stmt_name)

        # Parse memory accesses
        temps = {}
        reads = {}
        writes = {}
        accesses = self._scop._memory_accesses[stmt_name]
        for ref, access in accesses.items():
            if access.kind != "write":
                continue

            ref = Value.canonicalize(ref)
            inputs = set(["_in"])
            outputs = set(["_out"])
            tasklet = state.add_tasklet(
                name=ref,
                inputs=inputs,
                outputs=outputs,
                code="_out = " + access.as_cpp() + ";",
                language=dace.dtypes.Language.CPP,
            )

            # Input
            arguments = access.arguments()
            arg = next(arguments.__iter__())
            node = self._generate_argument(state, arg, temps, reads)
            state.add_edge(
                node, None, tasklet, "_in", dace.Memlet(data=node.data, expr=None)
            )

            # Output
            if access.array not in writes:
                writes[access.array] = state.add_access(access.array)
                self._outputs.add(access.array)

            state.add_edge(tasklet, "_out", writes[access.array], None, access.memlet())
            temps[access.reference] = tasklet

        return state, state

    def _generate_argument(
        self,
        state: dace.SDFGState,
        value: Value,
        temps: Dict[str, dace.nodes.AccessNode],
        reads: Dict[str, dace.nodes.AccessNode],
    ) -> dace.nodes.AccessNode:
        if value.reference in temps:
            return temps[value.reference]

        ref = Value.canonicalize(value.reference)
        inputs = set([Value.canonicalize(arg.reference) for arg in value.arguments()])
        outputs = set(["_out"])
        tasklet = state.add_tasklet(
            name=ref,
            inputs=inputs,
            outputs=outputs,
            code="_out = " + value.as_cpp() + ";",
            language=dace.dtypes.Language.CPP,
        )
        if isinstance(value, Access):
            if value.array not in reads:
                node = state.add_access(value.array)
                reads[value.array] = node
                self._inputs.add(value.array)

            tasklet.add_in_connector("_in")
            state.add_edge(reads[value.array], None, tasklet, "_in", value.memlet())
        elif isinstance(value, Indirection):
            if value.array not in reads:
                node = state.add_access(value.array)
                reads[value.array] = node
                self._inputs.add(value.array)

            tasklet.add_in_connector("_in")
            state.add_edge(
                reads[value.array],
                None,
                tasklet,
                "_in",
                value.memlet(self._sdfg.arrays[value.array]),
            )

            for arg in value.arguments():
                node = self._generate_argument(state, arg, temps, reads)
                state.add_edge(
                    node,
                    None,
                    tasklet,
                    Value.canonicalize(arg.reference),
                    dace.Memlet(data=node.data, expr=None),
                )
        else:
            for arg in value.arguments():
                node = self._generate_argument(state, arg, temps, reads)
                state.add_edge(
                    node,
                    None,
                    tasklet,
                    Value.canonicalize(arg.reference),
                    dace.Memlet(data=node.data, expr=None),
                )

        temp, _ = self._sdfg.add_scalar(
            name="temp" + Value.canonicalize(value.reference),
            find_new_name=True,
            dtype=value.dtype,
            transient=True,
        )
        access_node = state.add_access(temp)
        state.add_edge(
            tasklet,
            "_out",
            access_node,
            None,
            dace.Memlet(data=temp, expr=None),
        )
        temps[value.reference] = access_node

        return access_node

    @staticmethod
    def generate(scop: Scop, optimize: bool = False) -> dace.SDFG:
        daisycache = Path() / ".daisycache"

        sdfg = dace.SDFG(Generator._sdfg_name(scop.name, scop.source))
        sdfg.build_folder = str(daisycache / sdfg.name / "dacecache")

        for _, param in scop._parameters.items():
            sdfg.add_symbol(param.name, param.dtype)

        for _, memref in scop._memrefs.items():
            if memref.kind == "array":
                sdfg.add_array(memref.name, memref.shape, memref.dtype, transient=False)
            elif memref.kind == "value":
                sdfg.add_scalar(memref.name, memref.dtype, transient=False)
            else:
                sdfg.add_scalar(memref.name, memref.dtype, transient=True)

        # Generation
        init_state = sdfg.add_state("init_state", is_start_state=True)
        generator = Generator(sdfg=sdfg, scop=scop)
        first_state, last_state = generator._visit(scop.ast, [], [])
        sdfg.add_edge(init_state, first_state, dace.InterstateEdge())

        dace.propagate_memlets_sdfg(sdfg)
        consolidate_edges(sdfg)
        sdfg.simplify()

        # Parallelization
        ## First round of auto-parallelization
        AutoParallelization.apply(sdfg)
        LoopSimplification.apply(sdfg)

        ## Second round of auto-parallelization
        AutoParallelization.apply(sdfg)
        LoopSimplification.apply(sdfg)

        ## Greedy loop fusion
        GreedyLoopFusion.apply(sdfg)
        sdfg.validate()

        # Shape inference
        shapes = infer_shape(scop, sdfg)
        symbol_mapping = {}
        for name, memref in scop._memrefs.items():
            if memref.kind != "array":
                continue

            for i, val in enumerate(memref.shape):
                if str(val) in sdfg.free_symbols:
                    dim = shapes[name][i][1]
                    symbol_mapping[str(val)] = dim

        sdfg.specialize(symbol_mapping)
        sdfg.simplify()

        # Validation
        Generator._validate(sdfg, scop)

        return sdfg

    @staticmethod
    def _validate(sdfg: dace.SDFG, scop: Scop) -> bool:
        arguments = []
        for name, memref in scop._memrefs.items():
            if memref.kind != "array":
                continue

            arguments.append(name)

        for name, memref in scop._memrefs.items():
            if memref.kind != "value":
                continue

            arguments.append(name)

        for _, param in scop._parameters.items():
            arguments.append(param.name)

        assert len(sdfg.arglist()) == len(arguments)
        for i, name in enumerate(sdfg.arglist().keys()):
            assert name == arguments[i]

    @staticmethod
    def _sdfg_name(scop_name: str, source: str) -> str:
        return (
            ("sdfg_" + source + "_" + scop_name)
            .replace(".", "")
            .replace("%", "")
            .replace("-", "_")
        )
