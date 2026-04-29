"""
Spherical-coordinate camera controller for tumble / pan / zoom.

Produces a USD row-vector 4x4 NumPy transform matrix that can be handed
directly to ovrtx.write_attribute(prim, "omni:xform", matrix).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SphericalCamera:
    """
    Camera positioned on a sphere around a target point.

    Angles are in degrees.  Isaac / SimReady stages are Z-up.

    Controls:
        orbit(dx, dy)  – tumble around the target (left-button drag)
        pan(dx, dy)    – move both eye and target (middle-button drag)
        zoom(delta)    – move eye along the view ray (scroll wheel)
        reset()        – return to default pose
    """

    azimuth:   float = 45.0    # horizontal angle around the stage up axis
    elevation: float = 30.0    # vertical angle above the ground plane
    radius:    float = 5.0     # distance from target
    target:    np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    up_axis:   str = "Z"

    # Sensitivity multipliers
    orbit_speed: float = 0.35
    look_speed:  float = 0.25
    pan_speed:   float = 0.005
    zoom_speed:  float = 0.22
    frame_margin: float = 4.0

    # ── Eye position ────────────────────────────────────────────────────────────

    @property
    def eye(self) -> np.ndarray:
        return self.target + self.radius * self._view_offset()

    # ── Controls ────────────────────────────────────────────────────────────────

    def orbit(self, dx: float, dy: float):
        """Rotate the eye around the target (tumble)."""
        self.azimuth  = (self.azimuth - dx * self.orbit_speed) % 360.0
        self.elevation = float(np.clip(
            self.elevation + dy * self.orbit_speed, -89.9, 89.9
        ))

    def look(self, dx: float, dy: float):
        """Rotate the camera view in place, as in Kit's RMB fly mode."""
        eye = self.eye.copy()
        self.azimuth = (self.azimuth - dx * self.look_speed) % 360.0
        self.elevation = float(np.clip(
            self.elevation + dy * self.look_speed, -89.9, 89.9
        ))
        self.target = eye - self.radius * self._view_offset()

    def pan(self, dx: float, dy: float):
        """Translate the target in camera space (pan)."""
        right, up, _forward = self._camera_axes()
        scale    = self.radius * self.pan_speed
        delta    = (-dx * right + dy * up) * scale
        self.target = self.target + delta

    def zoom(self, delta: float):
        """Dolly the camera along the view ray."""
        factor = math.exp(-delta * self.zoom_speed)
        self.radius = float(np.clip(self.radius * factor, 0.01, 1.0e7))

    def fly(self, right_amount: float, up_amount: float, forward_amount: float):
        """Move the camera target in local camera axes while preserving heading."""
        right, up, forward = self._camera_axes()
        distance = max(self.radius * 0.08, 0.05)
        delta = (right * right_amount + up * up_amount + forward * forward_amount) * distance
        self.target = self.target + delta

    def reset(self):
        """Return to the default pose."""
        self.azimuth   = 45.0
        self.elevation = 30.0
        self.radius    = 5.0
        self.target    = np.array([0.0, 0.0, 0.0])

    def frame_bounds(self, center: np.ndarray, extent: float):
        """
        Point the camera at `center` and pull back so that a sphere of
        radius `extent` fills the view.
        """
        self.target  = center.copy()
        self.radius  = max(extent * self.frame_margin, 3.0)

    def _view_offset(self) -> np.ndarray:
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        if self._normalized_up_axis() == "Z":
            return np.array([
                math.cos(el) * math.cos(az),
                math.cos(el) * math.sin(az),
                math.sin(el),
            ])
        return np.array([
            math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az),
        ])

    def _normalized_up_axis(self) -> str:
        return "Z" if str(self.up_axis).upper() == "Z" else "Y"

    # ── Transform matrix ────────────────────────────────────────────────────────

    def _camera_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return camera right, up, and forward axes in world space.

        USD cameras look down local -Z, so the transform stores camera back
        as +Z while navigation uses forward.
        """
        eye      = self.eye
        forward  = self.target - eye
        norm_f   = np.linalg.norm(forward)
        if norm_f < 1e-9:
            forward = np.array([0.0, 0.0, -1.0])
        else:
            forward /= norm_f

        if self._normalized_up_axis() == "Z":
            world_up = np.array([0.0, 0.0, 1.0])
            fallback_up = np.array([0.0, 1.0, 0.0])
        else:
            world_up = np.array([0.0, 1.0, 0.0])
            fallback_up = np.array([1.0, 0.0, 0.0])
        right    = np.cross(forward, world_up)
        rn       = np.linalg.norm(right)
        if rn < 1e-9:
            world_up = fallback_up
            right    = np.cross(forward, world_up)
            rn       = np.linalg.norm(right)
        right /= rn

        up = np.cross(right, forward)
        return right, up, forward

    def get_transform(self) -> np.ndarray:
        """
        Return a 4x4 camera-to-world transform in USD row-vector convention.

        OVRTX/GfMatrix4d expects basis vectors in rows and translation in
        row 3: matrix[3][0..2].
        """
        right, up, forward = self._camera_axes()
        eye = self.eye

        m = np.eye(4, dtype=np.float64)
        m[0, :3] = right
        m[1, :3] = up
        m[2, :3] = -forward   # camera looks down -Z
        m[3, :3] = eye
        return m

    def get_view_matrix(self) -> np.ndarray:
        """
        Return the inverse of get_transform() - the world-to-camera matrix.
        Useful for passing to legacy OpenGL code.
        """
        c2w = self.get_transform()
        r   = c2w[:3, :3]
        t   = c2w[3, :3]
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = r.T
        w2c[3, :3] = -t @ r.T
        return w2c
