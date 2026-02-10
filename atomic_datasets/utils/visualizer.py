from typing import Optional, Sequence, Union

import numpy as np

from atomic_datasets import datatypes

try:
    import py3Dmol
except ImportError:
    raise ImportError(
        "py3Dmol is required for visualization. "
        "Install with: pip install py3Dmol"
    )


def _to_xyz(positions: np.ndarray, atom_types: Sequence[str]) -> str:
    """Convert positions and atom types to an XYZ format string."""
    n = len(positions)
    lines = [str(n), ""]
    for atom, pos in zip(atom_types, positions):
        lines.append(f"{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    return "\n".join(lines)


def visualize(
    graph: datatypes.Graph,
    **kwargs: Union[str, int, bool, "py3Dmol.view"],
):
    return visualize_raw(
        positions=graph["nodes"]["positions"],
        atom_types=graph["nodes"]["atom_types"],
        **kwargs,
    )


def visualize_raw(
    positions: np.ndarray,
    atom_types: np.ndarray,
    style: str = "sphere+stick",
    width: int = 400,
    height: int = 400,
    background: str = "#ffffff",
    spin: bool = False,
    show_labels: bool = True,
    viewer: Optional["py3Dmol.view"] = None,
) -> "py3Dmol.view":
    """Visualize a molecule as an interactive 3D widget.

    Args:
        style: Drawing style — "sphere", "stick", "sphere+stick".
        infer_bonds: If True, automatically infer bonds based on distances.
        width: Viewer width in pixels.
        height: Viewer height in pixels.
        background: Background color (hex string or color name).
        spin: If True, the molecule rotates continuously.
        show_labels: If True, show element labels on each atom.
        viewer: Existing py3Dmol viewer to add to. If None, creates a new one.

    Returns:
        The py3Dmol viewer object (displays automatically in Jupyter).
    """

    positions = np.asarray(positions, dtype=np.float64)
    atom_types = [str(a) for a in atom_types]

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {positions.shape}")
    if len(atom_types) != len(positions):
        raise ValueError(
            f"Length mismatch: {len(positions)} positions, {len(atom_types)} atom_types"
        )

    if viewer is None:
        viewer = py3Dmol.view(width=width, height=height)

    viewer.setBackgroundColor(background)

    xyz_str = _to_xyz(positions, atom_types)
    viewer.addModel(xyz_str, "xyz")

    style_spec = {style: {}}
    if style == "sphere":
        style_spec = {"sphere": {"scale": 0.3}}
    elif style == "stick":
        style_spec = {"stick": {"radius": 0.15}}
    elif style == "sphere+stick":
        style_spec = {"sphere": {"scale": 0.25}, "stick": {"radius": 0.1}}
    else:
        raise ValueError(f"Unsupported style '{style}'. Use 'sphere', 'stick', or 'sphere+stick'.")

    viewer.setStyle(style_spec)

    if show_labels:
        for i, atom in enumerate(atom_types):
            viewer.addLabel(
                atom,
                {
                    "position": {
                        "x": float(positions[i, 0]),
                        "y": float(positions[i, 1]),
                        "z": float(positions[i, 2]),
                    },
                    "fontSize": 10,
                    "backgroundOpacity": 0.3,
                },
            )

    if spin:
        viewer.spin(True)

    viewer.zoomTo()
    return viewer