from typing import Optional, Tuple, Union

import numpy as np
from scipy import linalg as spla  # type: ignore

from pycufsm.types import Forces, Sect_Geom, Sect_Props

# Originally developed for MATLAB by Benjamin Schafer PhD et al
# Ported to Python by Brooks Smith MEng, PE
#
# Each function within this file was originally its own separate file.
# Original MATLAB comments, especially those retaining to authorship or
# change history, have been generally retained unaltered


def yield_mp(nodes: np.ndarray, f_y: float, sect_props: Sect_Props, restrained: bool = False) -> Forces:
    """Determine yield strengths in bending and axial loading

    Args:
        nodes (np.ndarray): _description_
        f_y (float): _description_
        sect_props (Sect_Props): _description_
        restrained (bool, optional): _description_. Defaults to False.

    Returns:
        forces (Forces): Yield bending and axial strengths
            {Py,Mxx_y,Mzz_y,M11_y,M22_y}

    BWS, Aug 2000
    BWS, May 2019 trap nan when flat plate or other properites are zero
    """
    f_yield: Forces = {"P": 0, "Mxx": 0, "Myy": 0, "M11": 0, "M22": 0, "restrain": restrained, "offset": [0, 0]}

    f_yield["P"] = f_y * sect_props["A"]

    # account for the possibility of restrained bending vs. unrestrained bending
    if restrained is False:
        sect_props["Ixy"] = 0
    # Calculate stress at every point based on m_xx=1
    m_xx = 1
    m_yy = 0
    stress1: np.ndarray = np.zeros((1, len(nodes)))
    stress1 = stress1 - (
        (m_yy * sect_props["Ixx"] + m_xx * sect_props["Ixy"]) * (nodes[:, 1] - sect_props["cx"])
        - (m_yy * sect_props["Ixy"] + m_xx * sect_props["Iyy"]) * (nodes[:, 2] - sect_props["cy"])
    ) / (sect_props["Iyy"] * sect_props["Ixx"] - sect_props["Ixy"] ** 2)
    if np.max(abs(stress1)) == 0:
        f_yield["Mxx"] = 0
    else:
        f_yield["Mxx"] = f_y / np.max(abs(stress1))
    # Calculate stress at every point based on m_yy=1
    m_xx = 0
    m_yy = 1
    stress1 = np.zeros((1, len(nodes)))
    stress1 = stress1 - (
        (m_yy * sect_props["Ixx"] + m_xx * sect_props["Ixy"]) * (nodes[:, 1] - sect_props["cx"])
        - (m_yy * sect_props["Ixy"] + m_xx * sect_props["Iyy"]) * (nodes[:, 2] - sect_props["cy"])
    ) / (sect_props["Iyy"] * sect_props["Ixx"] - sect_props["Ixy"] ** 2)
    if np.max(abs(stress1)) == 0:
        f_yield["Myy"] = 0
    else:
        f_yield["Myy"] = f_y / np.max(abs(stress1))
    # %M11_y, M22_y
    # %transform coordinates of nodes into principal coordinates
    phi = sect_props["phi"]
    transform = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    cent_coord = np.array([nodes[:, 1] - sect_props["cx"], nodes[:, 2] - sect_props["cy"]])
    prin_coord = np.transpose(spla.inv(transform) @ cent_coord)
    f_yield["M11"] = 1
    stress1 = np.zeros((1, len(nodes)))
    stress1 = stress1 - f_yield["M11"] * prin_coord[:, 1] / sect_props["I11"]
    if np.max(abs(stress1)) == 0:
        f_yield["M11"] = 0
    else:
        f_yield["M11"] = f_y / np.max(abs(stress1)) * f_yield["M11"]

    f_yield["M22"] = 1
    stress1 = np.zeros((1, len(nodes)))
    stress1 = stress1 - f_yield["M22"] * prin_coord[:, 0] / sect_props["I22"]
    if np.max(abs(stress1)) == 0:
        f_yield["M22"] = 0
    else:
        f_yield["M22"] = f_y / np.max(abs(stress1)) * f_yield["M22"]
    return f_yield


def stress_gen(
    nodes: np.ndarray,
    forces: Forces,
    sect_props: Sect_Props,
    restrained: bool = False,
    offset_basis: Union[int, list] = 0,
) -> np.ndarray:
    """Generates stresses on nodes based upon applied loadings

    Args:
        nodes (np.ndarray): _description_
        forces (Forces): _description_
        sect_props (Sect_Props): _description_
        restrained (bool, optional): _description_. Defaults to False.
        offset_basis (Union[int, list], optional): offset_basis compensates for section properties
            that are based upon coordinate
            [0, 0] being something other than the centreline of elements. For example,
            if section properties are based upon the outer perimeter, then
            offset_basis=[-thickness/2, -thickness/2]. Defaults to 0.

    Returns:
        np.ndarray: _description_

    BWS, 1998
    B Smith, Aug 2020
    """
    if "restrain" in forces:
        restrained = forces["restrain"]
    if "offset" in forces and forces["offset"] is not None:
        offset_basis = list(forces["offset"])
    if isinstance(offset_basis, (float, int)):
        offset_basis = [offset_basis, offset_basis]

    stress: np.ndarray = np.zeros((1, len(nodes)))
    stress = stress + forces["P"] / sect_props["A"]
    if restrained:
        stress = stress - (
            (forces["Myy"] * sect_props["Ixx"]) * (nodes[:, 1] - sect_props["cx"] - offset_basis[0])
            - (forces["Mxx"] * sect_props["Iyy"]) * (nodes[:, 2] - sect_props["cy"] - offset_basis[1])
        ) / (sect_props["Iyy"] * sect_props["Ixx"])
    else:
        stress = stress - (
            (forces["Myy"] * sect_props["Ixx"] + forces["Mxx"] * sect_props["Ixy"])
            * (nodes[:, 1] - sect_props["cx"] - offset_basis[0])
            - (forces["Myy"] * sect_props["Ixy"] + forces["Mxx"] * sect_props["Iyy"])
            * (nodes[:, 2] - sect_props["cy"] - offset_basis[1])
        ) / (sect_props["Iyy"] * sect_props["Ixx"] - sect_props["Ixy"] ** 2)
    phi = sect_props["phi"] * np.pi / 180
    transform = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    cent_coord = np.array(
        [nodes[:, 1] - sect_props["cx"] - offset_basis[0], nodes[:, 2] - sect_props["cy"] - offset_basis[1]]
    )
    prin_coord = np.transpose(spla.inv(transform) @ cent_coord)
    stress = stress - forces["M11"] * prin_coord[:, 1] / sect_props["I11"]

    stress = stress - forces["M22"] * prin_coord[:, 0] / sect_props["I22"]
    nodes[:, 7] = stress.flatten()
    return nodes
