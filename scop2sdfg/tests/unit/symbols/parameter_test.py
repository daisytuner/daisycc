from scop2sdfg.scop.symbols.parameter import Parameter


def test_construct():
    param = Parameter("%1", "i64", "p1")

    assert param.validate()
    assert len(param.arguments()) == 0
    assert param.as_cpp() == "p1"


def test_floating_point():
    try:
        _ = Parameter(
            "%1",
            "float",
            "p1",
        )
    except AssertionError:
        pass
