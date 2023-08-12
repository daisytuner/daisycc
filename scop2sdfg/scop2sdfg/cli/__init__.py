import dace
import shutil
import fire
import traceback

from pathlib import Path

from dace.transformation.auto.auto_optimize import make_transients_persistent

from daisytuner.optimization import Optimization

from scop2sdfg.scop.scop import Scop
from scop2sdfg.codegen.generator import Generator


class CLI(object):
    def __call__(
        self,
        source_path: str,
        scop: str,
        schedule: str = "sequential",
        transfer_tune: bool = False,
        topk: int = 3,
        use_profiling_features: bool = False,
    ):
        assert schedule in ["sequential", "multicore", "gpu"]

        source_path = Path(source_path)
        daisycache = Path() / ".daisycache"

        try:
            scop = Scop.from_json(source_path.name, scop)
            scop.validate()

            sdfg = Generator.generate(scop)
        except:
            traceback.print_exc()
            exit(1)

        # Prune unprofitable
        has_loop = sdfg.has_cycles()
        if not has_loop:
            for state in sdfg.states():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.MapEntry):
                        has_loop = True
                        break

                if has_loop:
                    break

        if not has_loop:
            exit(1)

        if schedule == "gpu":
            sdfg.apply_gpu_transformations()

        #  Tune and compile
        if transfer_tune and schedule != "gpu":
            report = Optimization.apply(
                sdfg=sdfg, topK=topk, use_profiling_features=use_profiling_features
            )

        if schedule == "sequential":
            for nsdfg in sdfg.all_sdfgs_recursive():
                for state in nsdfg.states():
                    for node in state.nodes():
                        if isinstance(node, dace.nodes.MapEntry):
                            node.map.schedule = dace.ScheduleType.Sequential

        # Set high-level schedule options
        sdfg.openmp_sections = False
        dace.sdfg.infer_types.infer_connector_types(sdfg)
        dace.sdfg.infer_types.set_default_schedule_and_storage_types(sdfg, None)
        make_transients_persistent(sdfg, dace.DeviceType.CPU)

        try:
            sdfg.compile()
            sdfg.save(daisycache / f"{sdfg.name}.sdfg")

            libname = "lib" + sdfg.name + ".so"
            shutil.copy(Path(sdfg.build_folder) / "build" / libname, daisycache)
        except:
            traceback.print_exc()
            exit(1)

        exit(0)


def main():
    fire.Fire(CLI)
