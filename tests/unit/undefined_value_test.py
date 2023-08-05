from daisytuner_llvm.scop.undefined_value import UndefinedValue


def test_is_valid():
    assert not UndefinedValue("%1", "double").validate()
