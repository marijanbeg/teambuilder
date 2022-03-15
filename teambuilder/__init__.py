import os
import pkg_resources
import pytest

from .teambuilder import TeamBuilder

__version__ = pkg_resources.get_distribution(__name__).version


def test():
    """Run all package tests.
    
    Examples
    --------
    1. Run all tests.
    
    >>> import teambuilder as tb
    ...
    >>> # tb.test()
    
    """
    
    return pytest.main(['-v', '--pyargs',
                        'teambuikder', '-l'])  # pragma: no cover