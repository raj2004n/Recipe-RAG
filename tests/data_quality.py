import pytest

# does not do any data quality checks currently
# TODO: add data quality checks

# code from geeksforgeeks
def func(x):
    return x+5

#defining an another function  
def test_method():
#check whether 3+5 = 8 or not by passing 3 as an argument in function x
    assert func(3) == 8