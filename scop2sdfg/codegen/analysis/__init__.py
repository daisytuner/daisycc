import re
import math
import dace
import islpy as isl


def infer_shape(scop, sdfg: dace.SDFG):
    # Hacky implementation
    ctx = isl.DEFAULT_CONTEXT

    alias_groups = scop._shape_inference
    bounds = {}
    if alias_groups is None:
        return bounds

    for alias_group in alias_groups:
        for member in alias_group["readwrite"]:
            name = re.search("MemRef\d+", member["minimal"]).group(0)
            if name not in bounds:
                bounds[name] = []

            minimal = isl.MultiPwAff.read_from_str(ctx, member["minimal"])
            for i in range(len(minimal)):
                if i >= len(bounds[name]):
                    bounds[name].append([float("inf"), float("-inf")])

                expr = minimal.get_at(i)
                expr = str(expr)
                expr = re.search("\d+", expr).group(0)
                bounds[name][i][0] = min(bounds[name][i][0], int(expr))

            maximal = isl.MultiPwAff.read_from_str(ctx, member["maximal"])
            for i in range(len(maximal)):
                expr = maximal.get_at(i)
                expr = str(expr)
                expr = re.search("\d+", expr).group(0)
                bounds[name][i][1] = max(bounds[name][i][1], int(expr))

        for member in alias_group["readonly"]:
            name = re.search("MemRef\d+", member["minimal"]).group(0)
            if name not in bounds:
                bounds[name] = []

            minimal = isl.MultiPwAff.read_from_str(ctx, member["minimal"])
            for i in range(len(minimal)):
                if i >= len(bounds[name]):
                    bounds[name].append([float("inf"), float("-inf")])

                expr = minimal.get_at(i)
                expr = str(expr)
                expr = re.search("\d+", expr).group(0)
                bounds[name][i][0] = min(bounds[name][i][0], int(expr))

            maximal = isl.MultiPwAff.read_from_str(ctx, member["maximal"])
            for i in range(len(maximal)):
                expr = maximal.get_at(i)
                expr = str(expr)
                expr = re.search("\d+", expr).group(0)
                bounds[name][i][1] = max(bounds[name][i][1], int(expr))

    for state in sdfg.states():
        for node in state.data_nodes():
            if not isinstance(sdfg.arrays[node.data], dace.data.Array):
                continue

            if node.data not in bounds:
                bounds[node.data] = [
                    [math.inf, -math.inf]
                    for _ in range(len(sdfg.arrays[node.data].shape))
                ]

            for edge in state.in_edges(node):
                memlet: dace.Memlet = edge.data
                for i, subset in enumerate(memlet.subset):
                    b, e, _ = subset
                    try:
                        bounds[node.data][i][0] = min(bounds[node.data][i][0], int(b))
                        bounds[node.data][i][1] = max(
                            bounds[node.data][i][1], int(e) + 1
                        )
                    except:
                        continue

            for edge in state.out_edges(node):
                memlet: dace.Memlet = edge.data
                for i, subset in enumerate(memlet.subset):
                    b, e, _ = subset
                    try:
                        bounds[node.data][i][0] = min(bounds[node.data][i][0], int(b))
                        bounds[node.data][i][1] = max(
                            bounds[node.data][i][1], int(e) + 1
                        )
                    except:
                        continue

    return bounds
