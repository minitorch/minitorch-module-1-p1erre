"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Any, Callable, Iterable, List

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x: A number of float type
        y: A number of float type

    """
    return x * y


# - id
def id(x: float) -> float:
    """Returns input unchanged"""
    return x


# - add
def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


# - neg
def neg(x: float) -> float:
    """Negates a number"""
    return -x


# - lt
def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another

    Args:
    ----
        x: a number of type float
        y: a number of type float

    Returns:
    -------
        true if a < b else false

    """
    return x < y


# - eq
def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal"""
    return x == y


# - max
def max(x: float, y: float) -> float:
    """Returns the larger of two numbers"""
    return x if x >= y else y


# - is_close
def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    """Checks if two numbers are close in value.

    Args:
    ----
        x: a number of type float
        y: a number of type float
        tol: a number of type float that differences of the numbers should be less than

    Return:
    ------
        true if abs(x-y) < tol

    """
    return x is not None and y is not None and abs(x - y) < tol


# - sigmoid


def sigmoid(x: float) -> float:
    """Calculates the sigmoid of a function."""
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


# - relu
def relu(x: float) -> float:
    """Applies the ReLu activation function"""
    return max(x, 0.0)


# - log
def log(x: float) -> float:
    """Caculates the natural logarithm"""
    return math.log(x)


# - exp
def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)


# - log_back
def log_back(x: float, y: float) -> float:
    """Calculates the derivative of the log times a second arg"""
    return y / x


# - inv
def inv(x: float) -> float:
    """Caculates the reciprocal"""
    return 1 / x


# - inv_back
def inv_back(x: float, y: float) -> float:
    """Computes the derivative of the reciprocal times a second arg"""
    return -y / (x**2)


# - relu_back
def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    return y * (1 if x > 0 else 0)


# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(fn: Callable[[Any], Any], ls: Iterable[Any]) -> Any:
    """Higher-order function that applies a given function to each element of a iterable"""
    for x in ls:
        yield fn(x)


# - zipWith
def zipWith(
    fn: Callable[
        [
            Any,
            Any,
        ],
        Any,
    ],
    xs: Iterable[Any],
    ys: Iterable[Any],
) -> Iterable[Any]:
    """Higher-order function that combines elements from two iterables using a given function"""
    ixs, iys = iter(xs), iter(ys)
    while True:
        try:
            a, b = next(ixs), next(iys)
            yield fn(a, b)
        except StopIteration:
            break


# - reduce
def reduce(fn: Callable[[Any, Any], Any], xs: Iterable[Any]) -> Any:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    if not xs:
        return None  # Should raise an exception here?
    ixs = iter(xs)
    x = next(ixs)
    for y in ixs:
        x = fn(x, y)
    return x


# Use these to implement
# - negList : negate a list
def negList(xs: List[float]) -> List[float]:
    """Negate all elements in a list using map"""
    return map(neg, xs)


# - addLists : add two lists together
def addLists(xs: List[float], ys: List[float]) -> List[float]:
    """Add corresponding elements from two lists using zipWith"""
    return list(zipWith(add, xs, ys))


# - sum: sum lists
def sum(xs: List[float]) -> float:
    """Sum all elements in a list using reduce"""
    return 0 if not xs else reduce(add, xs)


# - prod: take the product of lists


def prod(xs: List[float]) -> float:
    """Calculate the product of all elements in a list using reduce"""
    return reduce(mul, xs)


# TODO: Implement for Task 0.3.
