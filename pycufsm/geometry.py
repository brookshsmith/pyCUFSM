import re
from typing import Literal, Optional

import numpy as np

ANGLE_TOLERANCE = np.radians(5)  # Tolerance for detecting a corner in radians


def mesh_nodes(
    centerline_coords: np.ndarray,
    corner_radius_deg: float,
    mesh_corner_deg: Optional[float] = 22.5,
    mesh_side_len: Optional[float] = 0.0,
):
    """
    Convert centerline coordinates to node coordinates for a cross-section. Note that all meshing parameters are interpreted as maximum values; meshing will always be performed evenly over each curved or straight segment. For example, if a straight segment is 100mm long and the mesh side length is set to 30mm, the actual mesh side length will be 25mm to ensure an even distribution of nodes.

    Args:
        centerline_coords (array_like): An (N, 2) array of (x, y) coordinates along the centerline of the cross-section.
        corner_radius (float): The radius of the corners of the cross-section.
        mesh_corner_deg (float, optional): The angle in degrees for meshing the corners. Defaults to 22.5 degrees. If None, corners will appear as though chamfered, with two points representing the start and end of the corner arc only.
        mesh_side_len (float, optional): The desired length of the mesh elements along the sides. Defaults to 0.0, and if set to this, it will be configured to a quarter of the maximum outer dimension. If None, no meshing will be performed.

    Returns:
        np.ndarray: An (M, 2) array of (x, y) coordinates for the nodes of the cross-section.

    """
    # Initialize parameters
    centerline_coords = np.array(centerline_coords)
    corner_radius = corner_radius_deg * np.pi / 180  # Convert corner radius from degrees to radians
    mesh_corner_rad = mesh_corner_deg * np.pi / 180  # Convert mesh corner angle from degrees to radians
    if mesh_side_len is not None and mesh_side_len <= 0.0:
        # Default mesh side length to the maximum outer dimension divided by 4
        max_dim = np.max(np.ptp(centerline_coords, axis=0))
        mesh_side_len = max_dim / 4

    nodes = [[centerline_coords[0, 0], centerline_coords[0, 1]]]

    # Iterate through centerline coordinates to create nodes
    prev_pt = centerline_coords[0]
    for curr_pt, next_pt in zip(centerline_coords[1:], centerline_coords[2:]):
        prev_angle = np.arctan2(curr_pt[1] - prev_pt[1], curr_pt[0] - prev_pt[0])
        next_angle = np.arctan2(next_pt[1] - curr_pt[1], next_pt[0] - curr_pt[0])
        corner_angle = next_angle - prev_angle
        corner_offset = 0.0
        if np.abs(corner_angle) > ANGLE_TOLERANCE:
            corner_offset = np.abs(corner_radius / np.tan(corner_angle / 2))

        # Add straight segment before corner
        prev_length = np.sqrt((curr_pt[0] - prev_pt[0]) ** 2 + (curr_pt[1] - prev_pt[1]) ** 2) - corner_offset
        num_segments = int(np.ceil(prev_length / mesh_side_len)) if mesh_side_len else 1
        dx = (curr_pt[0] - prev_pt[0]) / num_segments
        dy = (curr_pt[1] - prev_pt[1]) / num_segments
        for i in range(1, num_segments + 1):
            nodes.append([prev_pt[0] + i * dx, prev_pt[1] + i * dy])

        # Add corner arc if needed
        if corner_offset > 0.0:
            centroid_x = curr_pt[0] + corner_radius * np.cos(prev_angle + np.sign(corner_angle) * np.pi / 2)
            centroid_y = curr_pt[1] + corner_radius * np.sin(prev_angle + np.sign(corner_angle) * np.pi / 2)
            start_angle = prev_angle - np.pi / 2
            num_segments = int(np.ceil(np.abs(corner_angle) / mesh_corner_rad)) if mesh_corner_rad else 1
            for i in range(1, num_segments + 1):
                theta = start_angle + i * (corner_angle / num_segments)
                arc_x = centroid_x + corner_radius * np.cos(theta)
                arc_y = centroid_y + corner_radius * np.sin(theta)
                nodes.append([arc_x, arc_y])

        prev_angle = next_angle
        prev_pt = nodes[-1]  # Which will equal curr_pt if no corner was added, else the last arc point added

    # Add final straight segment
    final_pt = next_pt
    final_length = np.sqrt((final_pt[0] - prev_pt[0]) ** 2 + (final_pt[1] - prev_pt[1]) ** 2)
    num_segments = int(np.ceil(final_length / mesh_side_len)) if mesh_side_len else 1
    dx = (final_pt[0] - prev_pt[0]) / num_segments
    dy = (final_pt[1] - prev_pt[1]) / num_segments
    for i in range(1, num_segments + 1):
        nodes.append([prev_pt[0] + i * dx, prev_pt[1] + i * dy])

    return np.array(nodes)


def c_section(
    b: float,
    d: float,
    l: float,
    t: float,
    b2: Optional[float] = None,
    r_inner: float = 0.0,
    mesh_corner_deg: Optional[float] = 22.5,
    mesh_side_len: Optional[float] = 0.0,
):
    """
    Convert cross-section outer dimensions of a "C" section to meshed nodes.

    Args:
        b (float): Width of the cross-section.
        d (float): Depth of the cross-section.
        l (float): Lip length of the cross-section.
        t (float): Thickness of the cross-section.
        b2 (float, optional): Width of the smaller flange of the cross-section. If present, will be placed at the top of the cross-section. If None, assumed to be equal to `b`.
        r_inner (float, optional): Inner radius of the cross-section corners. Defaults to 0.0.
        mesh_corner_deg (float, optional): The angle in degrees for meshing the corners. Defaults to 22.5 degrees. If None, corners will appear as though chamfered, with two points representing the start and end of the corner arc only.
        mesh_side_len (float, optional): The desired length of the mesh elements along the sides. Defaults to 0.0, and if set to this, it will be configured to a quarter of the maximum outer dimension. If None, no meshing will be performed.

    Returns:
        np.ndarray: An (N, 2) array of (x, y) coordinates along the centerline of the cross-section. Assumes (0,0) is at the lower left corner.
    """
    if b2 is None:
        b2 = b

    if l <= t:
        # 2_____3
        # |
        # |
        # |
        # 1_____0
        coords = [
            [b - t, 0],
            [0, 0],
            [0, d - t],
            [b2 - t, d - t],
        ]
    else:
        # 3_____4
        # |     5
        # |
        # |     0
        # 2_____1
        coords = [
            [b - t, l - t / 2],
            [b - t, 0],
            [0, 0],
            [0, d - t],
            [b2 - t, d - t],
            [b2 - t, d - (l - t / 2)],
        ]

    return mesh_nodes(
        centerline_coords=coords,
        corner_radius_deg=r_inner + t / 2,
        mesh_corner_deg=mesh_corner_deg,
        mesh_side_len=mesh_side_len,
    )


def z_section(
    b: float,
    d: float,
    l: float,
    t: float,
    b2: Optional[float] = None,
    r_inner: float = 0.0,
    mesh_corner_deg: Optional[float] = 22.5,
    mesh_side_len: Optional[float] = 0.0,
):
    """
    Convert cross-section outer dimensions of a "Z" section to centerline coordinates. This function does not mesh the cross-section, and does not account for corner radii. If these are needed, run this function's output through `centerline_coords_to_nodes()`.

    Args:
        b (float): Width of the cross-section.
        d (float): Depth of the cross-section.
        l (float): Lip length of the cross-section.
        t (float): Thickness of the cross-section.
        b2 (float, optional): Width of the smaller flange of the cross-section. If present, will be placed at the top of the cross-section. If None, assumed to be equal to `b`.
        r_inner (float, optional): Inner radius of the cross-section corners. Defaults to 0.0.
        mesh_corner_deg (float, optional): The angle in degrees for meshing the corners. Defaults to 22.5 degrees. If None, corners will appear as though chamfered, with two points representing the start and end of the corner arc only.
        mesh_side_len (float, optional): The desired length of the mesh elements along the sides. Defaults to 0.0, and if set to this, it will be configured to a quarter of the maximum outer dimension. If None, no meshing will be performed.

    Returns:
        np.ndarray: An (N, 2) array of (x, y) coordinates along the centerline of the cross-section. Assumes (0,0) is at the lower left corner.
    """
    if b2 is None:
        b2 = b

    if l <= t:
        # 3_____2
        #       |
        #       |
        #       |
        #       1_____0
        coords = [
            [b2 - t + b + t, 0],
            [b2 - t, 0],
            [b2 - t, d - t],
            [0, d - t],
        ]
    else:
        # 4_____3
        # 5     |
        #       |
        #       |     0
        #       2_____1
        coords = [
            [b2 - t + b - t, l - t / 2],
            [b2 - t + b - t, 0],
            [b2 - t, 0],
            [b2 - t, d - t],
            [0, d - t],
            [0, d - (l - t / 2)],
        ]

    return mesh_nodes(
        centerline_coords=coords,
        corner_radius_deg=r_inner + t / 2,
        mesh_corner_deg=mesh_corner_deg,
        mesh_side_len=mesh_side_len,
    )


def f_section(
    b: float,
    d: float,
    l: float,
    t: float,
    r_inner: float = 0.0,
    mesh_corner_deg: Optional[float] = 22.5,
    mesh_side_len: Optional[float] = 0.0,
):
    """
    Convert cross-section outer dimensions of a "F" section to centerline coordinates. This function does not mesh the cross-section, and does not account for corner radii. If these are needed, run this function's output through `centerline_coords_to_nodes()`.

    Args:
        b (float): Width of the cross-section.
        d (float): Depth of the cross-section.
        l (float): Lip length of the cross-section.
        t (float): Thickness of the cross-section.
        r_inner (float, optional): Inner radius of the cross-section corners. Defaults to 0.0.
        mesh_corner_deg (float, optional): The angle in degrees for meshing the corners. Defaults to 22.5 degrees. If None, corners will appear as though chamfered, with two points representing the start and end of the corner arc only.
        mesh_side_len (float, optional): The desired length of the mesh elements along the sides. Defaults to 0.0, and if set to this, it will be configured to a quarter of the maximum outer dimension. If None, no meshing will be performed.

    Returns:
        np.ndarray: An (N, 2) array of (x, y) coordinates along the centerline of the cross-section. Assumes (0,0) is at the lower left corner.
    """
    if d < 1:
        b_slope = 0.2344
    else:
        b_slope = 0.2188

    #       2_____3
    #      /       \
    # 0___1         4___5
    coords = [
        [0, 0],
        [l - t / 2, 0],
        [l + b_slope - t / 2, d - t],
        [l + b_slope + b - t, d - t],
        [l + 2 * b_slope + b - t, 0],
        [2 * l + 2 * b_slope + b - t, 0],
    ]

    return mesh_nodes(
        centerline_coords=coords,
        corner_radius_deg=r_inner + t / 2,
        mesh_corner_deg=mesh_corner_deg,
        mesh_side_len=mesh_side_len,
    )


def _sfia_thickness_and_radius(designation: int, thickness_type: Literal["minimum", "design"] = "design") -> dict:
    """
    Look up the actual thickness and corner radius for SFIA cold-formed steel sections based on designation thickness in mils.

    Args:
        designation (int): The SFIA designation thickness in mils (e.g., 18, 27, 30, etc.).
        thickness_type (str, optional): Type of thickness to use, either "minimum" or "design". Defaults to "design".

    Returns:
        dict: A dictionary containing:
            - t (float): The thickness based on the specified thickness type.
            - r (float): The centerline radius of the section.
            - t_min (float): The minimum thickness of the section.
            - t_design (float): The design thickness of the section.
            - r_inner (float): The inner radius of the section.
            - reference_gauge (str): The reference gauge for the designation.

    Raises:
        ValueError: If the designation is not found in the thickness table or if the thickness_type is not "minimum" or "design".
    """

    if thickness_type not in ["minimum", "design"]:
        raise ValueError(f"thickness_type must be either 'minimum' or 'design', got '{thickness_type}'.")

    # Source = January 2026 SFIA "Technical Guide for Cold-Formed Steel Framing Products"
    # https://www.cfsteel.org/sfia-technical-publications
    thickness_table = {
        18: {"t_min": 0.0179, "t_design": 0.0188, "r_inner": 0.0844, "reference_gauge": "25"},
        27: {"t_min": 0.0269, "t_design": 0.0283, "r_inner": 0.0796, "reference_gauge": "22"},
        30: {"t_min": 0.0296, "t_design": 0.0312, "r_inner": 0.0782, "reference_gauge": "20-Drywall"},
        33: {"t_min": 0.0329, "t_design": 0.0346, "r_inner": 0.0765, "reference_gauge": "20-Structural"},
        43: {"t_min": 0.0428, "t_design": 0.0451, "r_inner": 0.0712, "reference_gauge": "18"},
        54: {"t_min": 0.0538, "t_design": 0.0566, "r_inner": 0.0849, "reference_gauge": "16"},
        68: {"t_min": 0.0677, "t_design": 0.0713, "r_inner": 0.1070, "reference_gauge": "14"},
        97: {"t_min": 0.0966, "t_design": 0.1017, "r_inner": 0.1526, "reference_gauge": "12"},
        118: {"t_min": 0.1180, "t_design": 0.1242, "r_inner": 0.1841, "reference_gauge": "10"},
    }

    if int(designation) not in thickness_table:
        raise ValueError(f"Designation {int(designation)} not found in thickness table.")

    row = thickness_table[int(designation)]
    t = row["t_min"] if thickness_type == "minimum" else row["t_design"]
    r = row["r_inner"] + t / 2  # Convert inner radius to centerline radius
    row["t"] = t
    row["r"] = r
    return row


def _sfia_lip_length(flange_width: int) -> float:
    """
    Look up the lip length for SFIA cold-formed steel sections based on flange width.

    Args:
        flange_width (int): The flange width in hundredths of an inch (e.g., 200 for 2.00 inches).

    Returns:
        float: The lip length in inches.
    """
    # Source = January 2026 SFIA "Technical Guide for Cold-Formed Steel Framing Products"
    # https://www.cfsteel.org/sfia-technical-publications
    lip_length_table = {
        125: 0.188,
        137: 0.375,
        162: 0.500,
        200: 0.625,
        250: 0.625,
        300: 0.625,
        350: 1.000,
    }

    if int(flange_width) not in lip_length_table:
        raise ValueError(f"Flange width {int(flange_width)} not found in lip length table.")
    return lip_length_table[int(flange_width)]


def sfia_section(
    designation: str, mesh_corner_deg: Optional[float] = 22.5, mesh_side_len: Optional[float] = 0.0
) -> np.ndarray:
    """
    Create a SFIA cold-formed steel section based on its designation. All length units are in inches. This function supports "C", "S", "T", "U", "Z", and "F" section types.

    Args:
        designation (str): The SFIA designation of the section (e.g., "362S200-43").

    Returns:
        np.ndarray: An (N, 2) array of (x, y) coordinates for the nodes of the cross-section.

    Raises:
        ValueError: If the designation format is invalid or if the section type is unknown.
    """
    try:
        depth, section_type, flange_width, thickness_mils = re.match(
            r"(\d+)([STUFCZ])(\d+)-(\d+)", designation
        ).groups()
    except AttributeError as exc:
        raise ValueError(f"Invalid SFIA designation format: '{designation}'.") from exc

    d = float(depth) / 100  # Convert depth to inches
    if (d - round(d)) in [0.12, 0.37, 0.62, 0.87]:
        d = d + 0.005  # Adjust depth for designation rounding rules

    b = float(flange_width) / 100  # Convert flange width to inches
    if (b - round(b)) in [0.12, 0.37, 0.62, 0.87]:
        b = b + 0.005  # Adjust flange width for designation rounding rules

    thickness_info = _sfia_thickness_and_radius(int(thickness_mils), thickness_type="design")
    t = thickness_info["t"]
    r_inner = thickness_info["r_inner"]

    l = 0.0
    if section_type in ["S", "C", "Z"]:
        l = _sfia_lip_length(int(flange_width))
    elif section_type == "F":
        l = 0.500

    if section_type in ["S", "T", "U", "C"]:
        return c_section(
            b=b, d=d, l=l, t=t, r_inner=r_inner, mesh_corner_deg=mesh_corner_deg, mesh_side_len=mesh_side_len
        )
    if section_type in ["Z"]:
        return z_section(
            b=b, d=d, l=l, t=t, r_inner=r_inner, mesh_corner_deg=mesh_corner_deg, mesh_side_len=mesh_side_len
        )
    if section_type in ["F"]:
        return f_section(
            b=b, d=d, l=l, t=t, r_inner=r_inner, mesh_corner_deg=mesh_corner_deg, mesh_side_len=mesh_side_len
        )

    raise ValueError(f"Unknown section type '{section_type}' in designation '{designation}'.")
