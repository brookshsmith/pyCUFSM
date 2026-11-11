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

    def context_warping_constant():
        # Centreline dimensions shared by the closed-form checks below
        depth, width, thick = 10.0, 4.0, 0.1

        def _channel():
            coord = np.array([[width, 0.0], [0.0, 0.0], [0.0, depth], [width, depth]])
            ends = np.array([[0, 1, thick], [1, 2, thick], [2, 3, thick]], dtype=float)
            return coord, ends

        def _subdivided(coord: np.ndarray, ends: np.ndarray, parts: int) -> tuple[np.ndarray, np.ndarray]:
            # Split each element into ``parts`` collinear elements
            new_coord = [coord[int(ends[0, 0])]]
            new_ends = []
            for start, end, t in ends:
                for k in range(1, parts + 1):
                    new_coord.append(coord[int(start)] + (coord[int(end)] - coord[int(start)]) * k / parts)
                    new_ends.append([len(new_coord) - 2, len(new_coord) - 1, t])
            return np.array(new_coord), np.array(new_ends, dtype=float)

        def it_matches_the_closed_form_value_for_a_plain_channel():
            # Cw = t b^3 h^2 (3b + 2h) / (12 (6b + h))
            coord, ends = _channel()
            sect = cutwp.prop2(coord.copy(), ends.copy())
            expected = thick * width**3 * depth**2 * (3 * width + 2 * depth) / (12 * (6 * width + depth))
            assert sect["Cw"] == approx(expected)
            # the shear centre, computed before Cw, is unaffected: e = 3b^2 / (h + 6b)
            assert sect["x0"] == approx(-3 * width**2 / (depth + 6 * width))

        def it_matches_the_closed_form_value_for_a_zed():
            # Point-symmetric section with a non-zero product of inertia:
            # Cw = t b^3 h^2 (b + 2h) / (12 (2b + h))
            coord = np.array([[width, 0.0], [0.0, 0.0], [0.0, depth], [-width, depth]])
            ends = np.array([[0, 1, thick], [1, 2, thick], [2, 3, thick]], dtype=float)
            sect = cutwp.prop2(coord.copy(), ends.copy())
            expected = thick * width**3 * depth**2 * (width + 2 * depth) / (12 * (2 * width + depth))
            assert sect["Cw"] == approx(expected)

        def it_does_not_depend_on_the_element_discretisation():
            coord, ends = _channel()
            reference = cutwp.prop2(coord.copy(), ends.copy())["Cw"]
            fine_coord, fine_ends = _subdivided(coord, ends, parts=5)
            assert cutwp.prop2(fine_coord.copy(), fine_ends.copy())["Cw"] == approx(reference)

        def it_does_not_depend_on_the_element_order():
            coord, ends = _channel()
            reference = cutwp.prop2(coord.copy(), ends.copy())["Cw"]
            reversed_ends = ends[::-1].copy()
            assert cutwp.prop2(coord.copy(), reversed_ends)["Cw"] == approx(reference)

        def it_returns_a_normalised_unit_warping():
            # The normalised unit warping integrates to zero over the section
            coord, ends = _channel()
            sect = cutwp.prop2(coord.copy(), ends.copy())
            wn_vals = sect["wn"]
            assert wn_vals is not None
            integral = sum(
                (wn_vals[int(s)] + wn_vals[int(e)]) / 2 * t * np.hypot(*(coord[int(e)] - coord[int(s)]))
                for s, e, t in ends
            )
            assert integral == approx(0.0, abs=1e-9)
            assert np.max(np.abs(wn_vals)) > 0
