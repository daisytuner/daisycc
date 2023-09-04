import dace
import shutil
import fire
import traceback
import sys

from pathlib import Path

from daisytuner.optimization import Optimization
from daisytuner.transformations import MapSchedule

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
            print("Prune SDFG", flush=True)
            sys.exit(1)

        # Disable OpenMP sections
        sdfg.openmp_sections = False

        if schedule == "gpu":
            sdfg.apply_gpu_transformations()
        elif schedule == "sequential":
            sdfg.apply_transformations_repeated(
                MapSchedule, options={"schedule_type": dace.ScheduleType.Sequential}
            )
        elif schedule == "multicore":
            if transfer_tune:
                report = Optimization.apply(
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
