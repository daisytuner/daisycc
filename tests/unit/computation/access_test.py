import dace
import islpy as isl

from scop2sdfg.scop.undefined_value import UndefinedValue
from scop2sdfg.scop.symbols.loop import Loop
from scop2sdfg.scop.symbols.constant import Constant
from scop2sdfg.scop.computation.access import Access
from scop2sdfg.scop.symbols.memref import Memref


def test_load():
    memrefs = {
        "MemRef3": Memref.from_json(
            {
                "kind": "array",
                "name": "MemRef3",
                "sizes": ["*", "1100"],
                "type": "double",
                "variable": "double %84",
            }
        )
    }
    domain = isl.UnionSet.read_from_str(
        isl.DEFAULT_CONTEXT,
        "{ Stmt2[i0, i1, i2] : 0 <= i0 <= 799 and 0 <= i1 <= 899 and 0 <= i2 <= 1099 }",
    )
    desc = {
        "access_instruction": "  %85 = load double, ptr %84, align 8, !tbaa !5",
        "kind": "read",
        "incoming_value": "",
        "relation": "{ Stmt2[i0, i1, i2] -> MemRef3[i2, i1] }",
    }
    loops = {"%1": Loop("%1", "i64", "c0", "i1"), "%2": Loop("%2", "i64", "c1", "i2")}
    access = Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs)

    assert access.reference == "%85"
    assert access.dtype == dace.float64
    assert len(access.undefined_symbols()) == 0
    assert access.validate()

    assert access.array == "MemRef3"
    assert ",".join([str(expr) for expr in access._expr]) == "c1,c0"

    assert access.as_cpp() == "_in"

    arguments = access.arguments()
    assert len(arguments) == 0


def test_load_offsets():
    memrefs = {
        "MemRef0": Memref.from_json(
            {
                "kind": "array",
                "name": "MemRef0",
                "sizes": ["*", "118", "118"],
                "type": "double",
                "variable": "double %41",
            }
        )
    }
    domain = isl.UnionSet.read_from_str(
        isl.DEFAULT_CONTEXT,
        "{ Stmt3[i0, i1, i2, i3] : 0 <= i0 <= 499 and 0 <= i1 <= 117 and 0 <= i2 <= 117 and 0 <= i3 <= 117 }",
    )
    desc = {
        "access_instruction": "  %42 = load double, ptr %41, align 8, !tbaa !5",
        "kind": "read",
        "incoming_value": "",
        "relation": "{ Stmt3[i0, i1, i2, i3] -> MemRef0[2 + i1, 1 + i2, 1 + i3] }",
    }
    loops = {
        "%1": Loop("%1", "i64", "c0", "i1"),
        "%2": Loop("%2", "i64", "c1", "i2"),
        "%3": Loop("%3", "i64", "c2", "i3"),
    }
    access = Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs)

    assert access.reference == "%42"
    assert access.dtype == dace.float64
    assert len(access.undefined_symbols()) == 0
    assert access.validate()

    assert access.array == "MemRef0"
    assert ",".join([str(expr) for expr in access._expr]) == "c0 + 2,c1 + 1,c2 + 1"

    assert access.as_cpp() == "_in"

    arguments = access.arguments()
    assert len(arguments) == 0


def test_store_value():
    memrefs = {
        "MemRef3": Memref.from_json(
            {
                "kind": "array",
                "name": "MemRef3",
                "sizes": ["*", "1200"],
                "type": "double",
                "variable": "double %66",
            }
        )
    }
    domain = isl.UnionSet.read_from_str(
        isl.DEFAULT_CONTEXT, "{ Stmt13[i0, i1] : 0 <= i0 <= 799 and 0 <= i1 <= 1199 }"
    )
    desc = {
        "access_instruction": "  store double %65, ptr %66, align 8, !tbaa !5",
        "incoming_value": "  %65 = fdiv double %64, 1.100000e+03",
        "kind": "write",
        "relation": "{ Stmt13[i0, i1] -> MemRef3[i0, i1] }",
    }
    loops = {"%1": Loop("%1", "i64", "c0", "i0"), "%2": Loop("%2", "i64", "c1", "i1")}
    access = Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs)

    assert (
        access.reference
        != Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs).reference
    )
    assert access.dtype == dace.float64
    assert len(access.undefined_symbols()) == 0
    assert not access.validate()

    assert access.array == "MemRef3"
    assert ",".join([str(expr) for expr in access._expr]) == "c0,c1"

    assert access.as_cpp() == "_in"

    arguments = access.arguments()
    assert len(arguments) == 1
    assert isinstance(list(arguments)[0], UndefinedValue)
    assert list(arguments)[0].reference == "%65"


def test_store_value_offsets():
    memrefs = {
        "MemRef3": Memref.from_json(
            {
                "kind": "array",
                "name": "MemRef3",
                "sizes": ["*", "1200"],
                "type": "double",
                "variable": "double %66",
            }
        )
    }
    domain = isl.UnionSet.read_from_str(
        isl.DEFAULT_CONTEXT, "{ Stmt13[i0, i1] : 0 <= i0 <= 799 and 0 <= i1 <= 1199 }"
    )
    desc = {
        "access_instruction": "  store double %65, ptr %66, align 8, !tbaa !5",
        "incoming_value": "  %65 = fdiv double %64, 1.100000e+03",
        "kind": "write",
        "relation": "{ Stmt13[i0, i1] -> MemRef3[i0 + 1, 2i1] }",
    }
    loops = {"%1": Loop("%1", "i64", "c0", "i0"), "%2": Loop("%2", "i64", "c1", "i1")}
    access = Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs)

    assert (
        access.reference
        != Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs).reference
    )
    assert access.dtype == dace.float64
    assert len(access.undefined_symbols()) == 0
    assert not access.validate()

    assert access.array == "MemRef3"
    assert ",".join([str(expr) for expr in access._expr]) == "c0 + 1,2*c1"

    assert access.as_cpp() == "_in"

    arguments = access.arguments()
    assert len(arguments) == 1
    assert isinstance(list(arguments)[0], UndefinedValue)
    assert list(arguments)[0].reference == "%65"


def test_store_constant():
    memrefs = {
        "MemRef3": Memref.from_json(
            {
                "kind": "array",
                "name": "MemRef3",
                "sizes": ["*", "1200"],
                "type": "double",
                "variable": "double %66",
            }
        )
    }
    domain = isl.UnionSet.read_from_str(
        isl.DEFAULT_CONTEXT, "{ Stmt13[i0, i1] : 0 <= i0 <= 799 and 0 <= i1 <= 1199 }"
    )
    desc = {
        "access_instruction": "  store double 0.000000e+00, ptr %66, align 8, !tbaa !5",
        "incoming_value": "double 0.000000e+00",
        "kind": "write",
        "relation": "{ Stmt13[i0, i1] -> MemRef3[i0, i1] }",
    }
    loops = {"%1": Loop("%1", "i64", "c0", "i0"), "%2": Loop("%2", "i64", "c1", "i1")}
    access = Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs)

    assert (
        access.reference
        != Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs).reference
    )
    assert access.dtype == dace.float64
    assert len(access.undefined_symbols()) == 0
    assert access.validate()

    assert access.array == "MemRef3"
    assert ",".join([str(expr) for expr in access._expr]) == "c0,c1"

    assert access.as_cpp() == "_in"

    arguments = access.arguments()
    assert len(arguments) == 1
    assert isinstance(list(arguments)[0], Constant)
    assert list(arguments)[0].as_cpp() == "0.000000e+00"


def test_phi():
    memrefs = {
        "MemRef4": Memref.from_json(
            {
                "kind": "array",
                "name": "MemRef4",
                "sizes": ["*", "1200"],
                "type": "double",
                "variable": "double %48",
            }
        )
    }
    domain = isl.UnionSet.read_from_str(
        isl.DEFAULT_CONTEXT, "{ Stmt7[i0, i1] : 0 <= i0 <= 799 and 0 <= i1 <= 1199 }"
    )
    desc = {
        "access_instruction": "  %104 = phi double [ %101, %97 ], [ %109, %102 ]",
        "incoming_value": "  %101 = fmul double %100, 1.200000e+00",
        "kind": "write",
        "relation": "{ Stmt7[i0, i1] -> MemRef4[i0, i1] }",
    }
    loops = {"%1": Loop("%1", "i64", "c0", "i0"), "%2": Loop("%2", "i64", "c1", "i1")}
    access = Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs)

    assert (
        access.reference
        != Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs).reference
    )
    assert access.dtype == dace.float64
    assert len(access.undefined_symbols()) == 0
    assert not access.validate()

    assert access.array == "MemRef4"
    assert ",".join([str(expr) for expr in access._expr]) == "c0,c1"

    assert access.as_cpp() == "_in"

    arguments = access.arguments()
    assert len(arguments) == 1
    assert isinstance(list(arguments)[0], UndefinedValue)
    assert list(arguments)[0].reference == "%101"


def test_domain():
    memrefs = {
        "MemRef1": Memref.from_json(
            {
                "kind": "array",
                "name": "MemRef1",
                "sizes": ["*", "1200"],
                "type": "double",
                "variable": "double %48",
            }
        )
    }
    domain = isl.UnionSet.read_from_str(
        isl.DEFAULT_CONTEXT, "{ Stmt1[i0, i1] : 0 <= i0 <= 499 and 0 <= i1 <= 1199 }"
    )
    desc = {
        "access_instruction": "  store double %47, ptr %48, align 8, !tbaa !5",
        "incoming_value": "  %47 = load double, ptr %44, align 8, !tbaa !5",
        "kind": "write",
        "relation": "{ Stmt1[i0, i1] -> MemRef1[o0, i1 - 1200o0] : -1199 + i1 <= 1200o0 <= i1 }",
    }
    loops = {"%1": Loop("%1", "i64", "c0", "i0"), "%2": Loop("%2", "i64", "c1", "i1")}
    access = Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs)

    assert len(access.undefined_symbols()) == 0
    assert not access.validate()


def test_load_value():
    memrefs = {
        "MemRef2": Memref.from_json(
            {
                "kind": "value",
                "name": "MemRef2",
                "sizes": [],
                "type": "double",
                "variable": "double %4",
            }
        )
    }
    domain = isl.UnionSet.read_from_str(
        isl.DEFAULT_CONTEXT,
        "{ Stmt2[i0, i1, i2] : 0 <= i0 <= 3 and 0 <= i1 <= 49 and 0 <= i2 <= 99 }",
    )
    desc = {
        "access_instruction": "",
        "incoming_value": "",
        "kind": "read",
        "relation": "{ Stmt2[i0, i1, i2] -> MemRef2[] }",
    }
    loops = {
        "%1": Loop("%1", "i64", "c0", "i0"),
        "%2": Loop("%2", "i64", "c1", "i1"),
        "%3": Loop("%3", "i64", "c2", "i2"),
    }
    access = Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs)

    assert len(access.undefined_symbols()) == 0
    assert access.validate()

    assert access.reference == "%4"
    assert access.dtype == dace.float64

    assert access.array == "MemRef2"
    assert str(access._expr) == "[0]"

    assert access.as_cpp() == "_in"
    assert len(access.arguments()) == 0


def test_write_value():
    memrefs = {
        "MemRef5": Memref.from_json(
            {
                "kind": "value",
                "name": "MemRef5",
                "sizes": [],
                "type": "double",
                "variable": "double %90",
            }
        )
    }
    domain = isl.UnionSet.read_from_str(
        isl.DEFAULT_CONTEXT,
        "{ Stmt1[i0, i1] : 0 <= i0 <= 100 and 0 <= i1 <= 199 }",
    )
    desc = {
        "access_instruction": "  %90 = fadd fast double %89, %33",
        "incoming_value": "  %90 = fadd fast double %89, %33",
        "kind": "write",
        "relation": "{ Stmt1[i0, i1] -> MemRef5[] }",
    }
    loops = {
        "%10": Loop("%10", "i64", "c1", "i0"),
        "%19": Loop("%19", "i64", "c2", "i1"),
    }
    access = Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs)

    assert len(access.undefined_symbols()) == 0
    assert not access.validate()

    assert access.reference != Access.from_json(
        desc, domain=domain, loops=loops, memrefs=memrefs
    )
    assert access.dtype == dace.float64

    assert access.array == "MemRef5"
    assert str(access._expr) == "[0]"

    assert access.as_cpp() == "_in"
    assert len(access.arguments()) == 1


def test_indirect_access():
    memrefs = {
        "MemRef1": Memref.from_json(
            {
                "kind": "array",
                "name": "MemRef1",
                "sizes": ["*"],
                "type": "double",
                "variable": "  %2 = alloca [256 x double], align 16",
            }
        )
    }
    domain = isl.UnionSet.read_from_str(
        isl.DEFAULT_CONTEXT,
        "{ Stmt0[i0] : 0 <= i0 <= 255 }",
    )
    loops = {
        "%5": Loop("%5", "i64", "c0", "i0"),
    }
    desc = {
        "access_instruction": "  %10 = load double, ptr %9, align 8, !tbaa !9",
        "incoming_value": "",
        "kind": "read",
        "relation": "{ Stmt0[i0] -> MemRef1[o0] }",
    }

    access = Access.from_json(desc, domain=domain, loops=loops, memrefs=memrefs)

    assert len(access.undefined_symbols()) == 1
    assert not access.validate()

    assert access.reference == "%10"
    assert access.dtype == dace.float64

    assert access.array == "MemRef1"
    assert str(access._expr) == "[o0]"

    assert access.as_cpp() == "_in"
    assert len(access.arguments()) == 0
