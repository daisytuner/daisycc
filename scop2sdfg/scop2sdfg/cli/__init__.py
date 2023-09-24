import dace
import copy
import math
import shutil
import fire
import traceback
import sys
import warnings

from pathlib import Path

from dace.sdfg.analysis.cutout import SDFGCutout

from daisytuner.optimization import Optimization, Normalization
from daisytuner.transformations import MapSchedule
from daisytuner.transformations.helpers import find_all_parent_maps_recursive

from scop2sdfg.ir.scop import Scop
from scop2sdfg.codegen.generator import Generator
from scop2sdfg.codegen.analysis import infer_shape


class CLI(object):
    def __call__(
        self,
        source_path: str,
        scop: str,
        schedule: str = "sequential",
        transfer_tune: bool = False,
        topk: int = 3,
        use_profiling_features: bool = False,
        dump_raw_maps: bool = False,
    ):
        assert schedule in ["sequential", "multicore", "gpu"]

        source_path = Path(source_path)
        daisycache = Path() / ".daisycache"

        try:
            scop = Scop.from_json(source_path.name, scop)
            scop.validate()
            sdfg = Generator.generate(scop)
            sdfg.openmp_sections = False

            # Normalization
            Normalization.apply(sdfg)
            if not Normalization.is_normalized(sdfg):
                warnings.warn(
                    "Normalization did not succeed. This might result in sub-optimal performance."
                )

            # Shape inference
            shapes = infer_shape(scop, sdfg)
            symbol_mapping = {}
            for name, memref in scop._memrefs.items():
                if memref.kind != "array":
                    continue

                for i, val in enumerate(memref.shape):
                    if str(val) in sdfg.free_symbols:
                        dim = shapes[name][i][1]
                        if dim == -math.inf:
                            if i == 0:
                                dim = 1
                            else:
                                continue

                        symbol_mapping[str(val)] = dim

            sdfg.specialize(symbol_mapping)
            sdfg.simplify()

            Generator.validate(sdfg, scop)
        except:
            traceback.print_exc()
            sys.exit(1)

        # Prune unprofitable
        has_loop = False
        for nsdfg in sdfg.all_sdfgs_recursive():
            if nsdfg.has_cycles():
                has_loop = True
                break

            for state in nsdfg.states():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.MapEntry):
                        has_loop = True
                        break

                if has_loop:
                    break

        if not has_loop:
            sys.exit(1)

        # Dump maps for tuning purposes
        dump_raw_maps_path = None
        if dump_raw_maps:
            dump_raw_maps_path = daisycache / "raw_maps"
            dump_raw_maps_path.mkdir(parents=True, exist_ok=True)

            for nsdfg in sdfg.all_sdfgs_recursive():
                for state in nsdfg.states():
                    for node in state.nodes():
                        if not isinstance(node, dace.nodes.MapEntry):
                            continue

                        if find_all_parent_maps_recursive(state, node):
                            continue

                        map_exit = state.exit_node(node)
                        subgraph_nodes = set(state.all_nodes_between(node, map_exit))
                        subgraph_nodes.add(node)
                        subgraph_nodes.add(map_exit)

                        for edge in state.in_edges(node):
                            subgraph_nodes.add(edge.src)
                        for edge in state.out_edges(map_exit):
                            subgraph_nodes.add(edge.dst)

                        subgraph_nodes = list(subgraph_nodes)
                        cutout = SDFGCutout.singlestate_cutout(
                            state,
                            *subgraph_nodes,
                            symbols_map=copy.copy(sdfg.constants),
                        )
                        cutout.name = "cutout_" + str(cutout.hash_sdfg()).replace(
                            "-", "_"
                        )

                        for sym, val in sdfg.constants.items():
                            if sym in cutout.free_symbols:
                                cutout.specialize({sym: val})

                        cutout.save(dump_raw_maps_path / f"{cutout.name}.sdfg")

        if schedule == "gpu":
            sdfg.apply_gpu_transformations()
        elif schedule == "sequential":
            sdfg.apply_transformations_repeated(
                MapSchedule, options={"schedule_type": dace.ScheduleType.Sequential}
            )
        elif schedule == "multicore":
            if transfer_tune:
                _ = Optimization.apply(
                    sdfg=sdfg, topK=topk, use_profiling_features=use_profiling_features
                )

        # Set high-level schedule options
        dace.sdfg.infer_types.infer_connector_types(sdfg)
        dace.sdfg.infer_types.set_default_schedule_and_storage_types(sdfg, None)

        try:
            sdfg.compile()
            sdfg.save(daisycache / f"{sdfg.name}.sdfg")

            libname = "lib" + sdfg.name + ".so"
            shutil.copy(Path(sdfg.build_folder) / "build" / libname, daisycache)
        except:
            traceback.print_exc()
            sys.exit(1)

        sys.exit(0)


def main():
    fire.Fire(CLI)
