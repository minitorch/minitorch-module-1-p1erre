from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    v_plus = vals[:arg] + (vals[arg] + epsilon,) + vals[arg+1:]
    v_minus = vals[:arg] + (vals[arg] - epsilon,) + vals[arg+1:] 
    return (f(*v_plus) - f(*v_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # We are assuming that the graph have NOT multiple edges between two nodes
    # TODO: Implement for Task 1.4.
    
    # this solution assumes that the computation graph is a DAG

    ordered_variables = []
    marked_variables = set()

    def visit(v: Variable):
        if v.unique_id in marked_variables: 
            return
        for p in v.parents:
            if not p.is_constant():
                visit(p)
        ordered_variables.insert(0, v)
        marked_variables.add(v.unique_id)

    visit(variable)

    return ordered_variables


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    ordered_vars = topological_sort(variable)
    assert ordered_vars[0].unique_id == variable.unique_id

    non_leaf_var_deriv = {}
    non_leaf_var_deriv[variable.unique_id] = deriv    

    for var in ordered_vars:

        if var.is_leaf():
            continue
        
        dvar = non_leaf_var_deriv[var.unique_id]
        parents = var.chain_rule(dvar)
        for pv, d in parents:
            if pv.is_leaf():
                pv.accumulate_derivative(d)
            elif pv.unique_id not in non_leaf_var_deriv.keys():
                non_leaf_var_deriv[pv.unique_id] = d
            else:
                non_leaf_var_deriv[pv.unique_id] += d

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
