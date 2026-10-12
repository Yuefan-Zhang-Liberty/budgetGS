"""COLMAP binary scene reader for Mip-NeRF 360 datasets (pure stdlib).

Reads ``cameras.bin`` / ``images.bin`` / ``points3D.bin`` and produces the
tensor-ready scene objects the trainer consumes: per-view intrinsics and
camera-to-world matrices, plus SfM points for splat initialisation. The module
is deliberately torch- and numpy-free so the parser can be tested against the
real datasets with the system python3 before the conda environment exists.

Camera models: SIMPLE_PINHOLE (0) and PINHOLE (1) carry no distortion and are
fully supported. SIMPLE_RADIAL (2), RADIAL (3), OPENCV (4) and
OPENCV_FISHEYE (5) are parsed and their distortion coefficients are surfaced
on the scene for the trainer to forward to ``rasterization``; the week-1
360_v2 scenes (bicycle/garden/room) are PINHOLE.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field

# COLMAP camera model ids -> (name, number of double params). Standard COLMAP
# conventions (model_converter / read_write_model.py):
#   0 SIMPLE_PINHOLE  [f, cx, cy]
#   1 PINHOLE         [fx, fy, cx, cy]
#   2 SIMPLE_RADIAL   [f, cx, cy, k]
#   3 RADIAL          [f, cx, cy, k1, k2]
#   4 OPENCV          [fx, fy, cx, cy, k1, k2, p1, p2]
#   5 OPENCV_FISHEYE  [fx, fy, cx, cy, k1, k2, k3, k4]
CAMERA_MODEL_PARAMS: dict[int, int] = {
    0: 3,
    1: 4,
    2: 4,
    3: 5,
    4: 8,
    5: 8,
}


def read_cameras_binary(path: str) -> dict[int, dict]:
    """Return ``{camera_id: {model, width, height, params}}`` from cameras.bin."""
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        cameras: dict[int, dict] = {}
        for _ in range(num):
            (cid, model, width, height) = struct.unpack("<iiQQ", f.read(24))
            n = CAMERA_MODEL_PARAMS[model]
            params = list(struct.unpack(f"<{n}d", f.read(8 * n)))
            cameras[cid] = {"model": model, "width": width, "height": height, "params": params}
    return cameras


def _read_cstring(f) -> str:
    """Read a NUL-terminated string (COLMAP images.bin stores names this way)."""
    chunks: list[bytes] = []
    while True:
        b = f.read(1)
        if not b or b == b"\x00":
            break
        chunks.append(b)
    return b"".join(chunks).decode("utf-8")


def read_images_binary(path: str) -> list[tuple[str, list[float], list[float], int]]:
    """Return ``[(name, qvec, tvec, camera_id)]`` from images.bin.

    ``qvec`` is the (qw, qx, qy, qz) rotation of the world-to-camera matrix,
    ``tvec`` the world origin expressed in camera coordinates.

    The 360_v2 release uses a non-standard extension (verified against
    ``bicycle/sparse/0/images.bin``): after the 64-byte header and the
    NUL-terminated image name, each record carries ``uint64 n_obs`` followed by
    ``n_obs`` 24-byte 2D observation units. We skip the blob -- the trainer only
    needs the header fields; positions come from ``points3D.bin``.
    """
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        images: list[tuple[str, list[float], list[float], int]] = []
        for _ in range(num):
            (_iid,) = struct.unpack("<i", f.read(4))
            qvec = list(struct.unpack("<dddd", f.read(32)))
            tvec = list(struct.unpack("<ddd", f.read(24)))
            (camera_id,) = struct.unpack("<i", f.read(4))
            name = _read_cstring(f)
            (n_obs,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n_obs)  # embedded per-image 2D observations
            images.append((name, qvec, tvec, camera_id))
    return images


def read_points3d_binary(path: str) -> tuple[list[list[float]], list[list[int]]]:
    """Return ``(xyz, rgb)`` point lists from points3D.bin.

    The 360_v2 release variant (verified against ``bicycle/sparse/0``) writes
    each record as ``int64 prefix + xyz(3 doubles) + rgb(3 bytes) +
    error(double) + uint64 track_len + track_len x (image_id, point2D_idx)``;
    the prefix is a bookkeeping field we skip. Parses to exact EOF on all
    54 275 points of the bicycle scene.
    """
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        xyz: list[list[float]] = []
        rgb: list[list[int]] = []
        for _ in range(num):
            f.read(8)  # record prefix (360_v2 variant)
            p = struct.unpack("<ddd", f.read(24))
            c = struct.unpack("<BBB", f.read(3))
            f.read(8)  # reprojection error
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.read(8 * track_len)  # (image_id, point2D_idx) pairs
            xyz.append(list(p))
            rgb.append(list(c))
    return xyz, rgb


def qvec_to_rotmat(qvec: list[float]) -> list[list[float]]:
    """Unit-quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix."""
    w, x, y, z = qvec
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def _mat4x4_from_rot_t(R: list[list[float]], t: list[float]) -> list[float]:
    """Compose a 4x4 camera-to-world matrix from a rotation and translation."""
    Rt = [[R[c][r] for c in range(3)] for r in range(3)]  # transpose
    c = [-sum(Rt[r][c] * t[c] for c in range(3)) for r in range(3)]
    return Rt[0] + [c[0]] + Rt[1] + [c[1]] + Rt[2] + [c[2]] + [0.0, 0.0, 0.0, 1.0]


def _matmul44(a: list[float], b: list[float]) -> list[float]:
    """Multiply two row-major 4x4 matrices."""
    out = [0.0] * 16
    for r in range(4):
        for c in range(4):
            out[4 * r + c] = sum(a[4 * r + k] * b[4 * k + c] for k in range(4))
    return out


def _center_and_scale(points: list[list[float]]) -> tuple[list[float], float]:
    """Return ``(center, scale)`` translating the points to the origin and
    mapping the median radius to 1."""
    n = len(points)
    center = [
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    ]
    dists = [
        math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2 + (p[2] - center[2]) ** 2)
        for p in points
    ]
    s = sorted(dists)
    m = len(s)
    med = s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2.0
    scale = 1.0 / med if med > 1e-9 else 1.0
    return center, scale


@dataclass
class ColmapScene:
    """A parsed 360_v2-style scene, ready to be turned into torch tensors."""

    data_dir: str
    factor: int
    width: int
    height: int
    image_paths: list[str] = field(default_factory=list)
    camtoworlds: list[list[float]] = field(default_factory=list)  # [N][16]
    Ks: list[list[float]] = field(default_factory=list)  # [N][9]
    radial_coeffs: list[list[float]] = field(default_factory=list)  # [N][k]
    tangential_coeffs: list[list[float]] = field(default_factory=list)  # [N][k]
    points: list[list[float]] = field(default_factory=list)  # [M][3]
    points_rgb: list[list[int]] = field(default_factory=list)  # [M][3]
    transform: list[float] = field(
        default_factory=lambda: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    )
    scene_scale: float = 1.0


def intrinsics_from_camera(cam: dict, factor: int) -> tuple[list[float], list[float], list[float]]:
    """Return ``(K9, radial, tangential)`` for one camera at the downscale factor.

    ``K9`` is the row-major 3x3 intrinsics with fx/fy/cx/cy divided by the
    factor; ``radial``/``tangential`` are the distortion coefficient lists
    (empty for the pinhole models).
    """
    model = cam["model"]
    p = cam["params"]
    if model == 0:  # SIMPLE_PINHOLE
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
        radial, tangential = [], []
    elif model == 1:  # PINHOLE
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        radial, tangential = [], []
    elif model == 2:  # SIMPLE_RADIAL
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
        radial = [p[3], 0.0]
        tangential = []
    elif model == 3:  # RADIAL
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
        radial = p[3:5]
        tangential = []
    elif model in (4, 5):  # OPENCV / OPENCV_FISHEYE
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        radial = p[4:6] if model == 4 else p[4:8]
        tangential = p[6:8] if model == 4 else []
    else:
        raise ValueError(f"unsupported COLMAP camera model id {model}")
    K9 = [fx / factor, 0.0, cx / factor, 0.0, fy / factor, cy / factor, 0.0, 0.0, 1.0]
    return K9, radial, tangential


def _resolve_image_path(data_dir: str, name: str, factor: int) -> str:
    """Map a COLMAP image name onto the downscaled ``images_{factor}`` folder."""
    base = os.path.basename(name)
    if factor == 1:
        return os.path.join(data_dir, name)
    if name.startswith("images/"):
        return os.path.join(data_dir, f"images_{factor}", name[len("images/"):])
    return os.path.join(data_dir, f"images_{factor}", base)


def load_360_scene(data_dir: str, factor: int = 2) -> ColmapScene:
    """Load a Mip-NeRF 360 scene (``sparse/0`` + ``images_{factor}``).

    Normalisation follows gsplat's convention: the scene is translated to the
    SfM-point centroid and scaled to unit median radius, and ``scene_scale``
    is returned so the trainer can scale the means learning rate.
    """
    sparse = os.path.join(data_dir, "sparse", "0")
    cameras = read_cameras_binary(os.path.join(sparse, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse, "images.bin"))
    points, points_rgb = read_points3d_binary(os.path.join(sparse, "points3D.bin"))
    if not points:
        raise ValueError(f"load_360_scene: no SfM points in {sparse}")

    center, scale = _center_and_scale(points)
    T = [
        scale, 0.0, 0.0, -scale * center[0],
        0.0, scale, 0.0, -scale * center[1],
        0.0, 0.0, scale, -scale * center[2],
        0.0, 0.0, 0.0, 1.0,
    ]

    scene = ColmapScene(data_dir=data_dir, factor=factor, width=0, height=0)
    first_cam = cameras[images[0][3]]
    scene.width = first_cam["width"] // factor
    scene.height = first_cam["height"] // factor
    for name, qvec, tvec, camera_id in images:
        cam = cameras[camera_id]
        K9, radial, tangential = intrinsics_from_camera(cam, factor)
        c2w = _matmul44(T, _mat4x4_from_rot_t(qvec_to_rotmat(qvec), tvec))
        scene.image_paths.append(_resolve_image_path(data_dir, name, factor))
        scene.camtoworlds.append(c2w)
        scene.Ks.append(K9)
        scene.radial_coeffs.append(radial)
        scene.tangential_coeffs.append(tangential)
    scene.points = [
        [scale * (p[0] - center[0]), scale * (p[1] - center[1]), scale * (p[2] - center[2])]
        for p in points
    ]
    scene.points_rgb = points_rgb
    scene.transform = T
    scene.scene_scale = scale

    missing = [p for p in scene.image_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"load_360_scene: {len(missing)} image files missing "
            f"(factor={factor}), e.g. {missing[0]!r}"
        )
    return scene
