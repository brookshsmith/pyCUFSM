import numpy as np
from pytest import approx

from pycufsm.pre import cutwp


def _independent_ixy(coord: np.ndarray, ends: np.ndarray) -> float:
    """Product of inertia by the standard thin-walled line-element formula.

    For each line element the contribution is
    ``(dx*dy/12 + (xm - xc)*(ym - yc)) * length * thickness`` -- i.e. the
    element's own product of inertia about its midpoint plus the parallel-axis
    term, both weighted by the element area. This mirrors how ``Ixx``/``Iyy``
    are formed in :func:`cutwp.prop2`.
    """
    x_mean, y_mean, x_diff, y_diff, length, thick = [], [], [], [], [], []
    for start, end, t in ends:
        s, e = int(start), int(end)
        x_mean.append((coord[s, 0] + coord[e, 0]) / 2)
        y_mean.append((coord[s, 1] + coord[e, 1]) / 2)
        x_diff.append(coord[e, 0] - coord[s, 0])
        y_diff.append(coord[e, 1] - coord[s, 1])
        length.append(np.hypot(coord[e, 0] - coord[s, 0], coord[e, 1] - coord[s, 1]))
        thick.append(t)
    x_mean, y_mean, x_diff, y_diff, length, thick = map(np.array, (x_mean, y_mean, x_diff, y_diff, length, thick))
    area = np.sum(length * thick)
    x_c = np.sum(length * thick * x_mean) / area
    y_c = np.sum(length * thick * y_mean) / area
    return float(np.sum((x_diff * y_diff / 12 + (x_mean - x_c) * (y_mean - y_c)) * length * thick))


def describe_prop2():
    def context_product_of_inertia():
        def it_matches_the_thin_walled_formula_for_inclined_elements():
            # A section with an inclined element, so the element's own
            # contribution ``dx*dy/12`` to the product of inertia is non-zero
            # and must be weighted by ``length * thickness`` (issue #128).
            coord = np.array([[0.0, 0.0], [4.0, 0.0], [6.0, 3.0]])
            ends = np.array([[0, 1, 0.1], [1, 2, 0.1]], dtype=float)
            sect = cutwp.prop2(coord.copy(), ends.copy())
            assert sect["Ixy"] == approx(_independent_ixy(coord, ends))

        def it_is_zero_for_a_section_symmetric_about_x():
            coord = np.array([[0.0, -2.0], [2.0, -2.0], [0.0, 2.0], [2.0, 2.0]])
            ends = np.array([[0, 1, 0.1], [2, 3, 0.1], [0, 2, 0.1]], dtype=float)
            sect = cutwp.prop2(coord.copy(), ends.copy())
            assert sect["Ixy"] == approx(0.0)
