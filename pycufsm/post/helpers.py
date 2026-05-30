import numpy as np


def gammait2(phi: float, disp_local: np.ndarray) -> np.ndarray:
    """transform local displacements into global displacements

    Args:
        phi (float): angle
        disp_local (np.ndarray): local displacements

    Returns:
        np.ndarray: global displacements

    BWS, 1998
    """
    gamma = np.array([[np.cos(phi), 0, -np.sin(phi)], [0, 1, 0], [np.sin(phi), 0, np.cos(phi)]])
    return np.dot(np.linalg.inv(gamma), disp_local)  # type: ignore


def shapef(links: int, disp: np.ndarray, length: float) -> np.ndarray:
    """Apply displacements using shape function

    Args:
        links (int): the number of additional line segments used to show the disp shape
        disp (np.ndarray): the vector of nodal displacements
        length (float): the actual length of the element

    Returns:
        np.ndarray: applied displacements

    BWS, 1998
    """
    inc = 1 / (links)
    x_disps = np.linspace(inc, 1 - inc, links - 1)
    disp_local = np.zeros((3, len(x_disps)))
    for i, x_d in enumerate(x_disps):
        n_1 = 1 - 3 * x_d * x_d + 2 * x_d * x_d * x_d
        n_2 = x_d * length * (1 - 2 * x_d + x_d**2)
        n_3 = 3 * x_d**2 - 2 * x_d**3
        n_4 = x_d * length * (x_d**2 - x_d)
        n_matrix = np.array(
            [[(1 - x_d), 0, x_d, 0, 0, 0, 0, 0], [0, (1 - x_d), 0, x_d, 0, 0, 0, 0], [0, 0, 0, 0, n_1, n_2, n_3, n_4]]
        )
        disp_local[:, i] = np.dot(n_matrix, disp).reshape(3)
    return disp_local
