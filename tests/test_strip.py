import numpy as np
from pytest import approx

from pycufsm.fsm import strip
from pycufsm.pre.cutwp import prop2


def _channel() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Coarse plain channel (centreline 10 x 4 x 0.1) under uniform compression."""
    depth, width, thick = 10.0, 4.0, 0.1
    xy = [(width, 0), (width / 2, 0), (0, 0), (0, depth / 2), (0, depth), (width / 2, depth), (width, depth)]
    nodes = np.array([[i, x, y, 1, 1, 1, 1, 1.0] for i, (x, y) in enumerate(xy)], dtype=float)
    elements = np.array([[k, k, k + 1, thick, 0] for k in range(len(xy) - 1)], dtype=float)
    props = np.array([[0, 29500.0, 29500.0, 0.3, 0.3, 29500.0 / 2.6]])
    sect_props = prop2(nodes[:, 1:3].copy(), elements[:, 1:4].copy())
    return props, nodes, elements, sect_props


def _run(lengths: list[float], m_all: np.ndarray, b_c: str = "S-S") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    props, nodes, elements, sect_props = _channel()
    gbt_con = {"glob": [0], "dist": [0], "local": [0], "other": [0], "o_space": 1, "norm": 0, "couple": 1, "orth": 2}
    return strip(
        props=props,
        nodes=nodes,
        elements=elements,
        lengths=np.array(lengths),
        springs=np.array([]),
        constraints=np.array([]),
        GBT_con=gbt_con,
        B_C=b_c,
        m_all=m_all,
        n_eigs=10,
        sect_props=sect_props,
    )


def describe_strip():
    def context_number_of_valid_modes_varies_between_lengths():
        # At L = 10 all 10 requested modes are kept; at L = 1000 some load factors
        # exceed the spurious-eigenvalue filter, so fewer modes remain.
        lengths = [10.0, 1000.0]

        def it_returns_results_padded_to_the_largest_mode_count():
            _, curve, shapes = _run(lengths, np.ones((len(lengths), 1)))
            _, curve_short, _ = _run([lengths[1]], np.ones((1, 1)))
            n_long = np.count_nonzero(curve_short[0])
            assert n_long < curve.shape[1]
            assert curve.shape == (2, 10)
            assert shapes.shape[:2] == (2, 10)
            assert np.all(curve[1, n_long:] == 0.0)
            assert np.all(shapes[1, n_long:, :] == 0.0)

        def it_matches_running_each_length_on_its_own():
            signature, curve, shapes = _run(lengths, np.ones((len(lengths), 1)))
            for i, length in enumerate(lengths):
                sig_i, curve_i, shapes_i = _run([length], np.ones((1, 1)))
                n_i = curve_i.shape[1]
                assert signature[i] == approx(sig_i[0])
                assert curve[i, :n_i] == approx(curve_i[0])
                assert shapes[i, :n_i, :] == approx(shapes_i[0])

    def context_several_longitudinal_terms():
        def it_keeps_all_the_dof_of_every_term():
            m_all = np.array([[1, 2, 3]] * 2, dtype=float)
            _, nodes, _, _ = _channel()
            _, curve, shapes = _run([100.0, 200.0], m_all, b_c="C-C")
            assert curve.shape[0] == 2
            assert shapes.shape[2] == 4 * len(nodes) * 3
