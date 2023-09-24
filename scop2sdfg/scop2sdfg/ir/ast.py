import islpy as isl
import dace

from typing import Dict


class ASTBuilder:

    deps = [None]
    iterators = {}
    dimensions = {}

    def create(
        self,
        statements: Dict,
        context: isl.Set,
        schedule: isl.UnionMap,
        dependencies: isl.UnionMap,
    ) -> isl.AstNode:
        ast = ASTBuilder._get_ast_from_schedule_map(dependencies, schedule, context)

        for stmt in statements:
            if stmt in ASTBuilder.iterators:
                statements[stmt]["iterators"] = ASTBuilder.iterators[stmt]
                statements[stmt]["dimensions"] = ASTBuilder.dimensions[stmt]

        return ast

    @staticmethod
    def _at_each_domain(node: isl.AstNode, build: isl.AstBuild):
        """
        Annotated each node in the AST with the domain and partial schedule
        """
        info = UserInfo()
        id = isl.Id.alloc(ctx=isl.AstBuild.get_ctx(build), name="", user=info)
        info.build = isl.AstBuild.copy(build)
        info.schedule = build.get_schedule()
        info.domain = info.schedule.domain()
        node.set_annotation(id)

        if node.get_type() == isl.ast_node_type.user:
            ast_expr = node.user_get_expr()
            stmt = ast_expr.get_op_arg(0).to_C_str()

            dimensions = list(info.domain.as_set().get_var_dict().keys())

            ASTBuilder.iterators[stmt] = []
            ASTBuilder.dimensions[stmt] = []
            for i in range(0, ast_expr.get_op_n_arg() - 1):
                loop_iter = ast_expr.get_op_arg(i + 1)
                if loop_iter.get_type() != isl.ast_expr_type.id:
                    continue

                dimension = dimensions[i]
                ASTBuilder.iterators[stmt].append(loop_iter.to_C_str())
                ASTBuilder.dimensions[stmt].append(dimension)

        return node

    @staticmethod
    def _before_each_for(build: isl.AstBuild):
        """
        Detection of parallel loops.
        This function is called for each for in depth-first pre-order.
        """

        # A (partial) schedule for the domains elements for which part of
        # the AST still needs to be generated in the current build.
        # The domain elements are mapped to those iterations of the loops
        # enclosing the current point of the AST generation inside which
        # the domain elements are executed.
        part_sched = build.get_schedule()
        info = UserInfo()

        # Test for parallelism
        info.is_parallel = ASTBuilder._is_parallel(part_sched, ASTBuilder.deps[0])
        info.build = isl.AstBuild.copy(build)
        info.schedule = part_sched
        info.domain = part_sched.domain()

        id = isl.Id.alloc(ctx=build.get_ctx(), name="", user=info)
        return id

    @staticmethod
    def _is_parallel(part_sched: isl.UnionMap, stmt_deps: isl.UnionMap) -> bool:
        """
        Check if the current scheduling dimension is parallel by verifying that
        the loop does not carry any dependencies.
        :param part_sched: A partial schedule
        :param stmt_deps: The dependencies between the statements
        :return True if current the scheduling dimension is parallel, else False
        """

        # translate the dependencies into time-space, by applying part_sched
        time_deps = stmt_deps.apply_range(part_sched).apply_domain(part_sched)

        # the loop is parallel, if there are no dependencies in time-space
        if time_deps.is_empty():
            return True

        time_deps = isl.Map.from_union_map(time_deps)
        time_deps = time_deps.flatten_domain().flatten_range()

        curr_dim = time_deps.dim(isl.dim_type.set) - 1
        # set all dimension in the time-space equal, except the current one:
        # if the distance in all outer dimensions is zero, then it
        # has to be zero in the current dimension as well to be parallel
        for i in range(curr_dim):
            time_deps = time_deps.equate(isl.dim_type.in_, i, isl.dim_type.out, i)

        # computes a delta set containing the differences between image
        # elements and corresponding domain elements in the time_deps.
        time_deltas = time_deps.deltas()

        # the loop is parallel, if there are no deltas in the time-space
        if time_deltas.is_empty():
            return True

        # The loop is parallel, if the distance is zero in the current dimension
        delta = time_deltas.plain_get_val_if_fixed(isl.dim_type.set, curr_dim)
        return delta.is_zero()

    @staticmethod
    def _get_annotation_build(ctx: isl.Set, deps: isl.UnionMap) -> isl.AstBuild:
        """
        helper function that return an isl.AstBuild
        """
        build = isl.AstBuild.from_context(ctx)
        # callback _at_each_domain will be called for each domain AST node
        build, _ = build.set_at_each_domain(ASTBuilder._at_each_domain)
        ASTBuilder.deps = [deps]
        # callback _before_each_for be called in depth-first pre-order
        build, _ = build.set_before_each_for(ASTBuilder._before_each_for)
        return build

    @staticmethod
    def _get_ast_from_schedule_map(
        deps: isl.UnionMap, schedule_map: isl.UnionMap, context: isl.Set
    ) -> isl.AstNode:
        ctx = schedule_map.get_ctx()
        ctx.set_ast_build_atomic_upper_bound(True)
        ctx.set_ast_build_detect_min_max(True)

        build = ASTBuilder._get_annotation_build(context, deps)
        root = build.node_from_schedule_map(schedule_map)
        return root


class UserInfo:
    def __init__(self):
        # Loops is parallel
        self.is_parallel = False
        self.build = None
        self.schedule = None
        self.domain = None
