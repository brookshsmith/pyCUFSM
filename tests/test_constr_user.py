import numpy as np

from pycufsm.solve.cfsm import constr_user


def _nodes(n_nodes, fixed=()):
    """CUFSM node rows [node#, x, z, xdof, zdof, ydof, qdof, stress], every DOF free unless
    listed in `fixed` as (node, column) with column 3..6."""
    nodes = np.array([[i, 10.0 * i, 0.0, 1, 1, 1, 1, 1.0] for i in range(n_nodes)], dtype=float)
    for node, col in fixed:
        nodes[node, col] = 0
    return nodes


def _free_rows(r_matrix):
    """The global DOFs the basis can move (rows with any non-zero entry)."""
    return set(np.flatnonzero(np.any(r_matrix != 0, axis=1)))


def test_fixed_dof_one_term():
    # 3 nodes, 12 DOFs; node 0's x DOF (global DOF 0) fixed: 11 free DOFs remain
    r = constr_user(_nodes(3, fixed=[(0, 3)]), np.array([]), np.array([1.0]))
    assert r.shape == (12, 11)
    assert _free_rows(r) == set(range(1, 12))  # the fixed DOF stays fixed, every other DOF free
    np.testing.assert_array_equal(r.T @ r, np.eye(11))  # one unit column per free DOF


def test_fixed_dofs_several_terms_are_block_diagonal():
    # node 1's y (local v, DOF 3) and theta (DOF 2*3 + 3 = 9) fixed, terms m = 1, 2
    r = constr_user(_nodes(3, fixed=[(1, 5), (1, 6)]), np.array([]), np.array([1.0, 2.0]))
    assert r.shape == (24, 20)
    assert _free_rows(r) == set(range(24)) - {3, 9, 12 + 3, 12 + 9}
    # no coupling between the terms' blocks
    assert not np.any(r[:12, 10:]) and not np.any(r[12:, :10])
    np.testing.assert_array_equal(r.T @ r, np.eye(20))


def test_master_slave_constraint():
    # u of node 2 (DOF 4) = 0.5 * u of node 0 (DOF 0): DOF 4 is eliminated, 11 remain
    constraints = np.array([[2, 0, 0.5, 0, 0]], dtype=float)
    r = constr_user(_nodes(3), constraints, np.array([1.0]))
    assert r.shape == (12, 11)
    displ = r @ np.ones(11)  # every remaining DOF = 1
    assert displ[4] == 0.5 * displ[0]
