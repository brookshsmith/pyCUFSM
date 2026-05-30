from typing import List, Tuple

import ipywidgets as widgets  # pylint: disable=import-error
import numpy as np

from pycufsm.post import plotters


def prevals() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Returns default material, node, element, spring, constraint, and flag values.

    Returns:
        Tuple of (props, nodes, elements, springs, constraints, flag).
    """
    springs = np.array([])
    constraints = np.array([])
    flag = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    props = np.array([[0, 29500, 29500, 0.3, 0.3, 29500 / (2 * (1 + 0.3))]])
    nodes = np.array(
        [
            [0, 2.5, 0.773, 1, 1, 1, 1, 1],
            [1, 2.5, 0, 1, 1, 1, 1, 1],
            [2, 1.25, 0, 1, 1, 1, 1, 1],
            [3, 0, 0, 1, 1, 1, 1, 1],
            [4, 0, 3, 1, 1, 1, 1, 1],
            [5, 0, 6, 1, 1, 1, 1, 1],
            [6, 0, 9, 1, 1, 1, 1, 1],
            [7, 1.25, 9, 1, 1, 1, 1, 1],
            [8, 2.5, 9, 1, 1, 1, 1, 1],
            [9, 2.5, 8.227, 1, 1, 1, 1, 1],
        ]
    )
    elements = np.array(
        [
            [0, 0, 1, 0.059, 0],
            [1, 1, 2, 0.059, 0],
            [2, 2, 3, 0.059, 0],
            [3, 3, 4, 0.059, 0],
            [4, 4, 5, 0.059, 0],
            [5, 5, 6, 0.059, 0],
            [6, 6, 7, 0.059, 0],
            [7, 7, 8, 0.059, 0],
            [8, 8, 9, 0.059, 0],
        ]
    )
    return props, nodes, elements, springs, constraints, flag


# def input(page, length_BC):
#     props, nodes, elements, springs, constraints, flag = prevals()
#     page, props, nodes, elements = PreProcess(m,n,e, props, nodes, elements, page, springs, constraints, flag)
#     children = [page, page]
#     tab = widgets.Tab()
#     tab.children = children
#     tab.set_title(0, 'Main Preprocessor')
#     tab.set_title(1, 'Lengths and Boundary Conditions')
#     return tab


class Preprocess:  # pylint: disable=too-many-instance-attributes
    """Widget-based preprocessor for cross-section geometry input in Jupyter notebooks."""

    def __init__(self):
        """Initializes all instance attributes to default values."""
        self.m = 0
        self.n = 0
        self.e = 0
        self.nodes = np.array([])
        self.elements = np.array([])
        self.props = np.array([])
        self.springs = np.array([])
        self.constraints = np.array([])
        self.flag = []
        self.mitems = []
        self.nitems = []
        self.eitems = []
        self.add_mat_btn = None
        self.del_mat_btn = None
        self.add_node_btn = None
        self.del_node_btn = None
        self.add_elem_btn = None
        self.del_elem_btn = None
        self.submit_btn = None
        self.bc_widget = None
        self.neigs = None
        self.rowm = None
        self.rnode = None
        self.relem = None
        self.rflag = None
        self.r_bc = None
        self.cs = None
        self.page = None
        self.row1 = None
        self.row = None
        self.row0 = None
        self.b_c = None
        self.flags = None

    def wprops(self, m: int) -> Tuple[widgets.VBox, widgets.Button, widgets.Button, list]:
        """Builds the material properties widget panel.

        Args:
            m: Number of materials.

        Returns:
            Tuple of (panel widget, add button, delete button, material item list).
        """
        self.m = m
        matTitle = widgets.Label(value="Material Properties")
        mattext = ["mat#", "Ex", "Ey", "vx", "vy", "G"]
        prop: List[widgets.GridBox] = [[] for i in range(self.m)]
        self.mitems: List[widgets.FloatText] = [[] for i in range(self.m)]
        self.add_mat_btn = widgets.Button(description="Add Material", layout=widgets.Layout(border="solid 1px black"))
        self.del_mat_btn = widgets.Button(
            description="Remove Material",
            layout=widgets.Layout(border="solid 1px black"),
        )
        matlabel = widgets.GridBox(
            [widgets.Label(value=mattext[j]) for j in range(6)],
            layout=widgets.Layout(grid_template_columns="repeat(6, 50px[col-start])"),
        )
        for i in range(self.m):
            if i < len(self.props):
                for j in range(6):
                    self.mitems[i].append(
                        widgets.FloatText(value=self.props[i, j], layout=widgets.Layout(width="70px"))
                    )
                prop[i] = widgets.GridBox(
                    self.mitems[i],
                    layout=widgets.Layout(grid_template_columns="repeat(6, 50px[col-start])"),
                )
            if i >= len(self.props):
                for j in range(6):
                    self.mitems[i].append(
                        widgets.FloatText(
                            value=self.props[i - 1, j],
                            layout=widgets.Layout(width="70px"),
                        )
                    )
                prop[i] = widgets.GridBox(
                    self.mitems[i],
                    layout=widgets.Layout(grid_template_columns="repeat(6, 50px[col-start])"),
                )
        matr = widgets.VBox([prop[j] for j in range(m)])
        brow = widgets.HBox([self.add_mat_btn, self.del_mat_btn])
        self.rowm = widgets.VBox(
            [matTitle, matlabel, matr, brow],
            layout=widgets.Layout(border="solid 1px black", width="35%"),
        )
        return self.rowm, self.add_mat_btn, self.del_mat_btn, self.mitems

    def wnodes(self, nodes, n):
        """Builds the nodes widget panel.

        Args:
            nodes: Array of node data.
            n: Number of nodes.

        Returns:
            Tuple of (panel widget, add button, delete button, node item list).
        """
        self.nodes = nodes
        nodTitle = widgets.Label(value="Nodes")
        nodetext = ["Node#", "x", "y", "xdof", "zdof", "ydof", "qdof", "stress"]
        node = [[] for i in range(self.n)]
        self.nitems = [[] for i in range(self.n)]
        self.add_node_btn = widgets.Button(description="Add Node", layout=widgets.Layout(border="solid 1px black"))
        self.del_node_btn = widgets.Button(description="Remove Node", layout=widgets.Layout(border="solid 1px black"))
        nlabel = widgets.HBox([widgets.Label(value=nodetext[j], layout=widgets.Layout(width="55px")) for j in range(8)])
        for i in range(self.n):
            if i < len(self.nodes):
                for j in range(8):
                    self.nitems[i].append(
                        widgets.FloatText(
                            value=self.nodes[i, j],
                            layout=widgets.Layout(width="57px", height="22px", font=("Helvetica", 8)),
                        )
                    )
                node[i] = widgets.HBox(self.nitems[i], layout=widgets.Layout(width="490px", height="30px"))
            if i >= len(self.nodes):
                for j in range(8):
                    self.nitems[i].append(
                        widgets.FloatText(
                            value=self.nodes[i - 1, j],
                            layout=widgets.Layout(width="57px", height="22px", font=("Helvetica", 8)),
                        )
                    )
                node[i] = widgets.HBox(self.nitems[i], layout=widgets.Layout(width="490px", height="30px"))
        noder0 = widgets.VBox([node[j] for j in range(n)])
        brow = widgets.HBox([self.add_node_btn, self.del_node_btn])
        self.rnode = widgets.VBox(
            [nodTitle, nlabel, noder0, brow],
            layout=widgets.Layout(border="solid 1px black"),
        )
        return self.rnode, self.add_node_btn, self.del_node_btn, self.nitems

    def welems(self, elements, e):
        """Builds the elements widget panel.

        Args:
            elements: Array of element data.
            e: Number of elements.

        Returns:
            Tuple of (panel widget, add button, delete button, element item list).
        """
        self.elements = elements
        elTitle = widgets.Label(value="Elements")
        elemtext = ["Element#", "Nodei", "Nodej", "t", "Mat#"]
        elem = [[] for i in range(self.e)]
        self.eitems = [[] for i in range(self.e)]
        self.add_elem_btn = widgets.Button(
            description="Add Element",
            layout=widgets.Layout(border="solid 1px black", width="120px"),
        )
        self.del_elem_btn = widgets.Button(
            description="Remove Element",
            layout=widgets.Layout(border="solid 1px black", width="120px"),
        )
        elabel = widgets.GridBox(
            [widgets.Label(value=elemtext[j]) for j in range(5)],
            layout=widgets.Layout(grid_template_columns="repeat(5, 50px[col-start])"),
        )
        for i in range(self.e):
            if i < len(self.elements):
                for j in range(5):
                    self.eitems[i].append(
                        widgets.FloatText(
                            value=self.elements[i, j],
                            layout=widgets.Layout(width="50px", height="22px"),
                        )
                    )
                elem[i] = widgets.HBox(self.eitems[i], layout=widgets.Layout(height="30px"))
            if i >= len(self.elements):
                for j in range(5):
                    if j < 3:
                        self.eitems[i].append(
                            widgets.FloatText(
                                value=self.elements[i - 1, j] + 1,
                                layout=widgets.Layout(width="50px", height="22px"),
                            )
                        )
                    if j >= 3:
                        self.eitems[i].append(
                            widgets.FloatText(
                                value=self.elements[i - 1, j],
                                layout=widgets.Layout(width="50px", height="22px"),
                            )
                        )
                elem[i] = widgets.GridBox(self.eitems[i], layout=widgets.Layout(height="30px"))
        elemr = widgets.VBox([elem[j] for j in range(e)])
        brow = widgets.HBox([self.add_elem_btn, self.del_elem_btn], layout=widgets.Layout(width="30%"))
        self.relem = widgets.VBox(
            [elTitle, elabel, elemr, brow],
            layout=widgets.Layout(border="solid 1px black"),
        )
        return self.relem, self.add_elem_btn, self.del_elem_btn, self.eitems

    def wflag(self, flag):
        """Builds the plot options widget panel.

        Args:
            flag: List of flag values (0 or 1) for each plot option.

        Returns:
            Tuple of (panel widget, submit button, flag checkbox list).
        """
        FlagTitle = widgets.Label(value="Plot Options")
        Flagtext = [
            "node",
            "elem",
            "mat",
            "stress",
            "stresspic",
            "coord",
            "constraints",
            "springs",
            "origin",
            "propaxis",
        ]
        self.flags = []
        for i, label in enumerate(Flagtext):
            self.flags.append(
                widgets.Checkbox(
                    description=label,
                    value=flag[i] == 1,
                    indent=False,
                    layout=widgets.Layout(width="150px"),
                )
            )
        flag0 = widgets.VBox([self.flags[j] for j in range(10)])
        self.submit_btn = widgets.Button(
            description="Plot",
            layout=widgets.Layout(border="solid 1px black", width="150px"),
        )
        self.rflag = widgets.VBox([FlagTitle, flag0, self.submit_btn])
        return self.rflag, self.submit_btn, self.flags

    def w_bound_cond(self):
        """Builds the boundary conditions widget panel.

        Returns:
            Tuple of (panel widget, dropdown widget, eigenvalue count widget).
        """
        self.b_c = ["S-S", "C-C", "S-C", "C-F", "C-G"]
        BCtext = [
            "simple-simple",
            "clamped-clamped",
            "simple-clamped",
            "clamped-free",
            "clamped-guided",
        ]
        self.bc_widget = widgets.Dropdown(
            options=[(BCtext[i], i) for i in range(len(BCtext))],
            value=0,
            description="Boundary Conditions",
        )
        self.neigs = widgets.IntText(value=20, description="Number of eignevalues")
        self.r_bc = widgets.VBox([self.bc_widget, self.neigs], layout=widgets.Layout(width="50%"))
        return self.r_bc, self.bc_widget, self.neigs

    def assemble(self):
        """Assembles all sub-panels into the main page widget and wires button callbacks."""
        self.row1 = widgets.HBox([self.cs, self.rflag])
        self.row = widgets.HBox([self.rnode, self.row1])
        self.row0 = widgets.HBox([self.rowm, self.r_bc], layout=widgets.Layout(width="100%"))
        self.add_mat_btn.on_click(self.add_material)
        self.add_node_btn.on_click(self.add_node)
        self.add_elem_btn.on_click(self.add_elem)
        self.submit_btn.on_click(self.submit)
        self.del_mat_btn.on_click(self.del_material)
        self.del_node_btn.on_click(self.del_node)
        self.del_elem_btn.on_click(self.del_elem)
        self.page.close()
        del self.page
        self.page = widgets.VBox([self.row0, self.row, self.relem])

    def add_material(self):
        """Adds a new material row to the materials panel."""
        self.m = self.m + 1
        self.props = [[] for i in range(self.m)]
        for i in range(self.m):
            for j in range(6):
                if i >= len(self.mitems):
                    self.props[i].append(self.mitems[len(self.mitems) - 1][j].value)
                else:
                    self.props[i].append(self.mitems[i][j].value)
        self.props = np.array(self.props)
        self.rowm, self.add_mat_btn, self.del_mat_btn, self.mitems = self.wprops(self.m)
        self.cs = widgets.Output()
        with self.cs:
            plotters.cross_sect(self.nodes, self.elements, self.springs, self.constraints, self.flag)
        self.assemble()

    def del_material(self):
        """Removes the last material row from the materials panel."""
        self.m = self.m - 1
        self.props = [[] for i in range(self.m)]
        for i in range(self.m):
            for j in range(6):
                self.props[i].append(self.mitems[i][j].value)
        self.props = np.array(self.props)
        self.rowm, self.add_mat_btn, self.del_mat_btn, self.mitems = self.wprops(self.m)
        self.cs = widgets.Output()
        with self.cs:
            plotters.cross_sect(self.nodes, self.elements, self.springs, self.constraints, self.flag)
        self.assemble()

    def add_node(self):
        """Adds a new node row to the nodes panel."""
        self.n = self.n + 1
        self.nodes = [[] for i in range(self.n)]
        for i in range(self.n):
            for j in range(8):
                if i >= len(self.nitems):
                    self.nodes[i].append(self.nitems[len(self.nitems) - 1][j].value)
                else:
                    self.nodes[i].append(self.nitems[i][j].value)
        self.nodes = np.array(self.nodes)
        self.rnode, self.add_node_btn, self.del_node_btn, self.nitems = self.wnodes(self.nodes, self.n)
        self.cs = widgets.Output()
        with self.cs:
            plotters.cross_sect(self.nodes, self.elements, self.springs, self.constraints, self.flag)
        self.assemble()

    def del_node(self):
        """Removes the last node row from the nodes panel."""
        self.n = self.n - 1
        if self.n == len(self.eitems):
            self.del_elem()
        self.nodes = [[] for i in range(self.n)]
        for i in range(self.n):
            for j in range(8):
                self.nodes[i].append(self.nitems[i][j].value)
        self.nodes = np.array(self.nodes)
        self.rnode, self.add_node_btn, self.del_node_btn, self.nitems = self.wnodes(self.nodes, self.n)
        self.cs = widgets.Output()
        with self.cs:
            plotters.cross_sect(self.nodes, self.elements, self.springs, self.constraints, self.flag)
        self.assemble()

    def add_elem(self):
        """Adds a new element row to the elements panel."""
        self.e = self.e + 1
        self.elements = [[] for i in range(self.e)]
        for i in range(self.e):
            for j in range(5):
                if i >= len(self.eitems):
                    self.elements[i].append(self.eitems[len(self.eitems) - 1][j].value)
                else:
                    self.elements[i].append(self.eitems[i][j].value)
        self.elements = np.array(self.elements)
        self.relem, self.add_elem_btn, self.del_elem_btn, self.eitems = self.welems(self.elements, self.e)
        self.cs = widgets.Output()
        with self.cs:
            plotters.cross_sect(self.nodes, self.elements, self.springs, self.constraints, self.flag)
        self.assemble()

    def del_elem(self):
        """Removes the last element row from the elements panel."""
        self.e = self.e - 1
        self.elements = [[] for i in range(self.e)]
        for i in range(self.e):
            for j in range(5):
                self.elements[i].append(self.eitems[i][j].value)
        self.elements = np.array(self.elements)
        self.relem, self.add_elem_btn, self.del_elem_btn, self.eitems = self.welems(self.elements, self.e)
        self.cs = widgets.Output()
        with self.cs:
            plotters.cross_sect(self.nodes, self.elements, self.springs, self.constraints, self.flag)
        self.assemble()

    def submit(self):
        """Reads all widget values, updates stored arrays, and redraws the cross-section."""
        self.props = [[] for i in range(self.m)]
        for i in range(self.m):
            for j in range(6):
                self.props[i].append(self.mitems[i][j].value)
        self.props = np.array(self.props)
        self.nodes = [[] for i in range(self.n)]
        for i in range(self.n):
            for j in range(8):
                self.nodes[i].append(self.nitems[i][j].value)
        self.nodes = np.array(self.nodes)
        self.elements = [[] for i in range(self.e)]
        for i in range(self.e):
            for j in range(5):
                self.elements[i].append(self.eitems[i][j].value)
        self.elements = np.array(self.elements)
        for i, flag_widget in enumerate(self.flags):
            self.flag[i] = 1 if flag_widget.value else 0
        self.rflag, self.submit_btn, self.flags = self.wflag(self.flag)
        self.cs = widgets.Output()
        with self.cs:
            plotters.cross_sect(self.nodes, self.elements, self.springs, self.constraints, self.flag)
        self.assemble()

    def run(self, m, n, e, props, nodes, elements, springs, constraints, flag):
        """Initializes and displays the full preprocessor UI.

        Args:
            m: Number of materials.
            n: Number of nodes.
            e: Number of elements.
            props: Array of material properties.
            nodes: Array of node data.
            elements: Array of element data.
            springs: Array of spring data.
            constraints: Array of constraint data.
            flag: List of plot option flags.

        Returns:
            Tuple of updated (props, nodes, elements).
        """
        self.m = m
        self.n = n
        self.e = e
        self.nodes = nodes
        self.elements = elements
        self.props = props
        self.springs = springs
        self.constraints = constraints
        self.flag = flag
        self.rowm, self.add_mat_btn, self.del_mat_btn, self.mitems = self.wprops(len(self.props))
        self.rnode, self.add_node_btn, self.del_node_btn, self.nitems = self.wnodes(self.nodes, len(self.nodes))
        self.relem, self.add_elem_btn, self.del_elem_btn, self.eitems = self.welems(self.elements, len(self.elements))
        self.rflag, self.submit_btn, self.flags = self.wflag(self.flag)
        self.r_bc, self.bc_widget, self.neigs = self.w_bound_cond()
        self.cs = widgets.Output()
        with self.cs:
            plotters.cross_sect(self.nodes, self.elements, self.springs, self.constraints, self.flag)
        self.page = widgets.FloatText(value=1)
        self.assemble()
        return self.props, self.nodes, self.elements
