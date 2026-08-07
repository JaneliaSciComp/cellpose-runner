import cellpose_runner


def test_version():
    assert isinstance(cellpose_runner.__version__, str)
