from scop2sdfg.scop.symbols.loop import Loop


def test_construct():
    loop = Loop("%1", "i64", "c1", "i0")

    assert loop.validate()
    assert len(loop.arguments()) == 0

    assert loop._shift is None
    assert loop.as_cpp() == "c1"


def test_floating_point():
    try:
        _ = Loop("%1", "float", "c1", "i0")
    except AssertionError:
        pass
