import dace

from scop2sdfg.scop.value import Value


def test_from_ctype():
    value = Value("%1", "double")
    assert value.reference == "%1"
    assert value.dtype == dace.float64


def test_from_dtype():
    value = Value("%1", dace.int32)
    assert value.reference == "%1"
    assert value.dtype == dace.int32


def test_is_llvm_value():
    assert Value.is_llvm_value("%1")
    assert Value.is_llvm_value("@1")

    assert not Value.is_llvm_value("1")
    assert not Value.is_llvm_value("1.0")
    assert not Value.is_llvm_value("1.0e-1")
    assert not Value.is_llvm_value(".1")
    assert not Value.is_llvm_value("true")
    assert not Value.is_llvm_value("0x1234567")


def test_canonicalize():
    assert Value.canonicalize("%1") == "_1"
    assert Value.canonicalize("@1") == "_1"
    assert Value.canonicalize(".lcsa") == "lcsa"
