from typing import Callable, Tuple

import pytest
import numpy as np
from hypothesis import given
from hypothesis.strategies import DrawFn, composite, floats, integers

import minitorch
from minitorch import (
    Scalar,
    central_difference,
)

from .strategies import assert_close


@composite
def scalars(
    draw: DrawFn, min_value: float = -100000, max_value: float = 100000
) -> Scalar:
    val = draw(floats(min_value=min_value, max_value=max_value))
    return minitorch.Scalar(val)


small_scalars = scalars(min_value=-100, max_value=100)

@composite
def random_scalars(draw: DrawFn) -> Scalar:
    # Generate a random float between -1 and 1 using the formula 2 * (random.random() - 0.5)
    val = draw(floats(min_value=-1.0, max_value=1.0))
    return minitorch.Scalar(val)

@composite
def binary_integers(draw: DrawFn) -> int:
    # Generate a random integer that is either 0 or 1
    val = draw(integers(min_value=0, max_value=1))
    return val

# 2 * (random.random() - 0.5)

@given(binary_integers(), small_scalars, random_scalars(), random_scalars(), random_scalars())
def test_simple(y: int, x0: Scalar, x1: Scalar, w0: Scalar, w1: Scalar) -> None:
    # Simple add
    def f(x0: Scalar, x1: Scalar, w0: Scalar, w1: Scalar) -> Scalar:
        z0 = x0 * w0
        z1 = z0 + x1 * w1
        z2 = z1
        out = z2.sigmoid()

        if y == 1:
            prob = out
        else:
            prob = -out + 1.0
        
        # Add small epsilon for numerical stability
        eps = minitorch.Scalar(1e-15)
        prob_safe = prob + eps
        return -prob_safe.log()
    
    scalar_vars = [x0, x1, w0, w1]
    out = f(*scalar_vars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, var in enumerate(scalar_vars):
        check = central_difference(f, *scalar_vars, arg=i)
        print(y, str([v.data for v in scalar_vars]), out, var.derivative, i, check)
        assert var.derivative is not None
        np.testing.assert_allclose(
            var.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([v.data for v in scalar_vars]), var.derivative, i, check.data),
        )
    

    
    #assert_close(c.data, a + b)

    # Add others if you would like...

@given(random_scalars())
def test_subtraction_simple(x: Scalar) -> None:
    """Test the simple subtraction function 1 - x"""
    def f(x: Scalar) -> Scalar:
        return minitorch.Scalar(1.0) - x
    
    # Compute autodiff gradient
    result = f(x)
    result.backward()
    
    # Compute central difference gradient
    check = central_difference(f, x, arg=0)
    
    print(f"x={x.data}, result={result.data}, autodiff_grad={x.derivative}, central_diff_grad={check.data}")
    
    # The derivative of (1 - x) with respect to x should be -1
    expected_grad = -1.0
    print(f"Expected gradient: {expected_grad}")
    
    assert x.derivative is not None
    np.testing.assert_allclose(
        x.derivative,
        check.data,
        1e-2,
        1e-2,
        err_msg=f"Autodiff gave {x.derivative}, central diff gave {check.data}"
    )

@given(random_scalars(), random_scalars(), random_scalars(), random_scalars())
def test_subtraction_with_sigmoid(x0: Scalar, x1: Scalar, w0: Scalar, w1: Scalar) -> None:
    """Test the exact computation from the failing test: 1.0 - sigmoid(x0*w0 + x1*w1)"""
    def f(x0: Scalar, x1: Scalar, w0: Scalar, w1: Scalar) -> Scalar:
        z0 = x0 * w0
        z1 = z0 + x1 * w1
        out = z1.sigmoid()
        return minitorch.Scalar(1.0) - out
    
    # Compute autodiff gradients
    scalar_vars = [x0, x1, w0, w1]
    result = f(*scalar_vars)
    result.backward()
    
    print(f"Input values: x0={x0.data}, x1={x1.data}, w0={w0.data}, w1={w1.data}")
    print(f"Result: {result.data}")
    
    # Check each gradient
    for i, var in enumerate(scalar_vars):
        check = central_difference(f, *scalar_vars, arg=i)
        print(f"Var {i}: autodiff_grad={var.derivative}, central_diff_grad={check.data}")
        
        assert var.derivative is not None
        np.testing.assert_allclose(
            var.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=f"Variable {i}: Autodiff gave {var.derivative}, central diff gave {check.data}"
        )