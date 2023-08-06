from scop2sdfg.scop.symbols.constant import Constant


def test_floating_point():
    const = Constant(Constant.new_identifier(), "double", "1.0")

    assert len(const.arguments()) == 0
    assert const.validate()
    assert const.as_cpp() == "1.0"


def test_integer():
    const = Constant(Constant.new_identifier(), "i8", "1")

    assert len(const.arguments()) == 0
    assert const.validate()
    assert const.as_cpp() == "1"


def test_bool():
    const = Constant(Constant.new_identifier(), "i1", "true")

    assert len(const.arguments()) == 0
    assert const.validate()
    assert const.as_cpp() == "true"


def test_variable():
    try:
        Constant(Constant.new_identifier(), "i1", "%1")
    except AssertionError:
        pass


def test_is_constant():
    # Variables
    assert not Constant.is_constant("%1")

    # Integer
    assert Constant.is_constant("1")

    # Bool
    assert Constant.is_constant("true")
    assert Constant.is_constant("false")

    # Float
    assert Constant.is_constant("1.0")
    assert Constant.is_constant("1.0e-1")
    assert Constant.is_constant(".1")
    assert Constant.is_constant("0x1234567")

    # Strings
    assert Constant.is_constant('"abc"')
