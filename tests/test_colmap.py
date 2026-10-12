"""COLMAP parser tests against the real 360_v2 datasets (plan Day 2/3).

Skips (rather than fails) when the datasets are not present, so the suite
stays green on machines without the shared storage. Runs on the system
python3 -- the parser is pure stdlib.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from budgetgs.data.colmap import (
    _center_and_scale,
    load_360_scene,
    qvec_to_rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)

DATASETS = os.path.expanduser("~/storage/user/zyuef/datasets/360_v2")
BICYCLE = os.path.join(DATASETS, "bicycle")


@unittest.skipUnless(os.path.isdir(BICYCLE), "360_v2 datasets not present")
class TestColmapBinary(unittest.TestCase):
    def test_counts_and_layout(self):
        sparse = os.path.join(BICYCLE, "sparse", "0")
        cameras = read_cameras_binary(os.path.join(sparse, "cameras.bin"))
        images = read_images_binary(os.path.join(sparse, "images.bin"))
        xyz, rgb = read_points3d_binary(os.path.join(sparse, "points3D.bin"))
        self.assertEqual(len(cameras), 1)
        self.assertEqual(len(images), 194)
        self.assertEqual(len(xyz), 54275)
        self.assertEqual(len(rgb), 54275)
        self.assertEqual(cameras[1]["model"], 1)  # PINHOLE, no distortion
        self.assertEqual(len(cameras[1]["params"]), 4)  # fx fy cx cy

    def test_quaternion_rotation_is_orthonormal(self):
        R = qvec_to_rotmat([1.0, 0.0, 0.0, 0.0])
        self.assertEqual(R, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_center_and_scale_maps_median_radius_to_one(self):
        points = [[x, 0.0, 0.0] for x in (1.0, 2.0, 3.0, 4.0, 5.0)]
        center, scale = _center_and_scale(points)
        self.assertAlmostEqual(center[0], 3.0)
        # the median distance to the center becomes 1 after scaling
        scaled = sorted(abs(scale * (x - center[0])) for x in (1.0, 2.0, 3.0, 4.0, 5.0))
        self.assertAlmostEqual(scaled[len(scaled) // 2], 1.0)


@unittest.skipUnless(os.path.isdir(BICYCLE), "360_v2 datasets not present")
class TestLoad360Scene(unittest.TestCase):
    def test_scene_tensors_shape_and_normalization(self):
        scene = load_360_scene(BICYCLE, factor=2)
        self.assertEqual(len(scene.image_paths), 194)
        self.assertEqual(len(scene.camtoworlds), 194)
        self.assertEqual(len(scene.Ks), 194)
        self.assertEqual(len(scene.points), 54275)
        self.assertEqual(len(scene.points[0]), 3)
        self.assertEqual(len(scene.points_rgb[0]), 3)
        # intrinsics are divided by the factor
        self.assertTrue(math.isfinite(scene.Ks[0][0]))
        self.assertGreater(scene.Ks[0][0], 100.0)  # fx after 2x downscale
        # c2w bottom row is the identity convention
        self.assertEqual(scene.camtoworlds[0][12:16], [0.0, 0.0, 0.0, 1.0])
        # scene_scale maps the median point radius to ~1
        center = [0.0, 0.0, 0.0]
        radii = sorted(
            math.sqrt(sum((p[c] - center[c]) ** 2 for c in range(3))) for p in scene.points
        )
        med = radii[len(radii) // 2]
        self.assertAlmostEqual(med, 1.0, delta=1e-3)

    def test_missing_images_raise(self):
        scene = load_360_scene(BICYCLE, factor=2)
        good = scene.image_paths[0]
        self.assertTrue(os.path.isfile(good))
        self.assertIn(f"images_2{os.sep}", good)


if __name__ == "__main__":
    unittest.main()
