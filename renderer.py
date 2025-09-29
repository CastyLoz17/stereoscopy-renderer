import tkinter as tk
from math import *
import time
from typing import Union, List, Dict, Tuple
import multiprocessing as mp

import sys
import os

import random

try:
    from PIL import Image, ImageTk

    def _generate_one_static_image(args):
        width, height, filename, color_range = args
        if os.path.exists(filename):
            return filename

        static_list = [
            tuple(random.randint(color_range[0], color_range[1]) for _ in range(3))
            for _ in range(width * height)
        ]

        img = Image.new("RGB", (width, height))
        img.putdata(static_list)
        img.save(filename)
        return filename

    def generate_static_images(width, height, count=10, name="static_{}", folder="bg"):

        folder_path = resource_path(folder)
        os.makedirs(folder_path, exist_ok=True)
        filenames = [
            os.path.join(folder_path, f"{name.format(i+1)}.png") for i in range(count)
        ]

        color_range = (10, 60)
        processes = max(1, os.cpu_count() // 2)

        args_list = [(width, height, fn, color_range) for fn in filenames]

        with mp.Pool(processes) as p:
            results = p.map(_generate_one_static_image, args_list)

        for i, fn in enumerate(results, 1):
            print(f"{i}/{count} static images generated: {fn}")

        return results

    using_pillow = True

except:
    using_pillow = False

Number = Union[int, float]


class Vector2:
    __slots__ = ["x", "y"]

    def __init__(self, x: Number, y: Number):
        self.x, self.y = float(x), float(y)

    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: Number) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: Number) -> "Vector2":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> "Vector2":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vector2":
        return Vector2(-self.x, -self.y)

    def __eq__(self, other: "Vector2") -> bool:
        epsilon = 1e-10
        return abs(self.x - other.x) < epsilon and abs(self.y - other.y) < epsilon

    def __hash__(self):
        return hash((round(self.x, 10), round(self.y, 10)))

    def dot(self, other: "Vector2") -> float:
        return self.x * other.x + self.y * other.y

    def magnitude(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y)

    def magnitude_squared(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalize(self) -> "Vector2":
        mag_sq = self.x * self.x + self.y * self.y
        if mag_sq == 0:
            return Vector2(0, 0)
        inv_mag = 1.0 / sqrt(mag_sq)
        return Vector2(self.x * inv_mag, self.y * inv_mag)

    def distance_to(self, other: "Vector2") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return sqrt(dx * dx + dy * dy)

    def angle(self) -> float:
        return atan2(self.y, self.x)

    def rotate(self, angle: float) -> "Vector2":
        cos_a, sin_a = cos(angle), sin(angle)
        return Vector2(self.x * cos_a - self.y * sin_a, self.x * sin_a + self.y * cos_a)

    def is_zero(self) -> bool:
        epsilon = 1e-10
        return abs(self.x) < epsilon and abs(self.y) < epsilon


class Vector3:
    __slots__ = ["x", "y", "z"]

    def __init__(self, x: Number, y: Number, z: Number):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: Number) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: Number) -> "Vector3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> "Vector3":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    def __eq__(self, other: "Vector3") -> bool:
        epsilon = 1e-10
        return (
            abs(self.x - other.x) < epsilon
            and abs(self.y - other.y) < epsilon
            and abs(self.z - other.z) < epsilon
        )

    def __hash__(self):
        return hash((round(self.x, 10), round(self.y, 10), round(self.z, 10)))

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def magnitude_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalize(self) -> "Vector3":
        mag_sq = self.x * self.x + self.y * self.y + self.z * self.z
        if mag_sq == 0:
            return Vector3(0, 0, 0)
        inv_mag = 1.0 / sqrt(mag_sq)
        return Vector3(self.x * inv_mag, self.y * inv_mag, self.z * inv_mag)

    def distance_to(self, other: "Vector3") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return sqrt(dx * dx + dy * dy + dz * dz)

    def is_zero(self) -> bool:
        epsilon = 1e-10
        return abs(self.x) < epsilon and abs(self.y) < epsilon and abs(self.z) < epsilon

    def project_onto(self, other: "Vector3") -> "Vector3":
        if other.is_zero():
            return Vector3(0, 0, 0)
        return other * (self.dot(other) / other.magnitude_squared())

    def reflect(self, normal: "Vector3") -> "Vector3":
        return self - 2 * self.project_onto(normal)

    def rotate_point_around_axis(self, anchor, axis, angle):
        return rotate_point_around_axis(self, anchor, axis, angle)


def zero2() -> Vector2:
    return Vector2(0, 0)


def zero3() -> Vector3:
    return Vector3(0, 0, 0)


def unit_x3() -> Vector3:
    return Vector3(1, 0, 0)


def unit_y3() -> Vector3:
    return Vector3(0, 1, 0)


def unit_z3() -> Vector3:
    return Vector3(0, 0, 1)


def triangulate_face(face):
    if len(face) < 3:
        return []
    elif len(face) == 3:
        return [face]
    elif len(face) == 4:
        return [[face[0], face[1], face[2]], [face[0], face[2], face[3]]]
    else:
        triangles = []
        for i in range(1, len(face) - 1):
            triangles.append([face[0], face[i], face[i + 1]])
        return triangles


def rotate_point_around_axis(point, anchor, axis, angle):
    p = point - anchor
    k = axis.normalize()

    cos_a = cos(angle)
    sin_a = sin(angle)

    rotated = (
        p * cos_a
        + Vector3(k.y * p.z - k.z * p.y, k.z * p.x - k.x * p.z, k.x * p.y - k.y * p.x)
        * sin_a
        + k * (k.dot(p)) * (1 - cos_a)
    )

    return rotated + anchor


def compute_newell_normal(face):
    normal_x = normal_y = normal_z = 0.0
    face_len = len(face)
    for i in range(face_len):
        v1 = face[i]
        v2 = face[(i + 1) % face_len]
        normal_x += (v1.y - v2.y) * (v1.z + v2.z)
        normal_y += (v1.z - v2.z) * (v1.x + v2.x)
        normal_z += (v1.x - v2.x) * (v1.y + v2.y)
    return Vector3(normal_x, normal_y, normal_z)


def compute_cross_product_normal(face):
    for i in range(len(face) - 2):
        for j in range(i + 1, len(face) - 1):
            for k in range(j + 1, len(face)):
                v0 = face[i]
                v1 = face[j]
                v2 = face[k]
                edge1 = v1 - v0
                edge2 = v2 - v0
                if (
                    edge1.magnitude_squared() < 1e-12
                    or edge2.magnitude_squared() < 1e-12
                ):
                    continue
                normal = edge1.cross(edge2)
                if normal.magnitude_squared() > 1e-12:
                    return normal.normalize()
                break
            else:
                continue
            break
        else:
            continue
        break
    return Vector3(0, 0, 1)


def calculate_face_normal(face):
    if len(face) < 3:
        return Vector3(0, 0, 1)

    normal = compute_newell_normal(face)

    if normal.magnitude_squared() > 1e-12:
        return normal.normalize()
    else:
        return compute_cross_product_normal(face)


def calculate_face_centroid(face):
    face_len = len(face)
    sum_x = sum_y = sum_z = 0.0
    for vertex in face:
        sum_x += vertex.x
        sum_y += vertex.y
        sum_z += vertex.z
    return Vector3(sum_x / face_len, sum_y / face_len, sum_z / face_len)


def ray_intersects_triangle(ray_origin, ray_direction, triangle):
    epsilon = 1e-10

    if len(triangle) < 3:
        return False, 0

    v0, v1, v2 = triangle[0], triangle[1], triangle[2]

    edge1 = v1 - v0
    edge2 = v2 - v0

    if edge1.magnitude_squared() < epsilon or edge2.magnitude_squared() < epsilon:
        return False, 0

    h = ray_direction.cross(edge2)
    a = edge1.dot(h)

    if abs(a) < epsilon:
        return False, 0

    f = 1.0 / a
    s = ray_origin - v0
    u = f * s.dot(h)

    if u < -epsilon or u > 1.0 + epsilon:
        return False, 0

    q = s.cross(edge1)
    v = f * ray_direction.dot(q)

    if v < -epsilon or u + v > 1.0 + epsilon:
        return False, 0

    t = f * edge2.dot(q)

    if t > epsilon:
        return True, t
    else:
        return False, 0


def is_point_occluded(light_pos, target_point, objects, current_object=None):
    light_direction = (target_point - light_pos).normalize()
    light_distance = (target_point - light_pos).magnitude()

    shadow_bias = 0.001

    for obj in objects:
        if current_object is not None and obj is current_object:
            continue

        for face, material in obj.faces:
            if len(face) < 3:
                continue

            face_center = sum(face, Vector3(0, 0, 0)) / len(face)
            face_normal = calculate_face_normal(face)

            light_to_face = (face_center - light_pos).normalize()
            if face_normal.dot(light_to_face) > -0.1:
                continue

            triangles = triangulate_face(face)

            for triangle in triangles:
                intersects, intersection_distance = ray_intersects_triangle(
                    light_pos, light_direction, triangle
                )

                if (
                    intersects
                    and intersection_distance > shadow_bias
                    and intersection_distance < light_distance - shadow_bias
                ):
                    return True

    return False


def clip_polygon_edge(polygon, edge_type, edge_value):
    if not polygon or len(polygon) < 3:
        return []

    result = []
    n = len(polygon)

    for i in range(n):
        current = polygon[i]
        next_vertex = polygon[(i + 1) % n]

        if edge_type == "left":
            current_inside = current[0] >= edge_value
            next_inside = next_vertex[0] >= edge_value
        elif edge_type == "right":
            current_inside = current[0] <= edge_value
            next_inside = next_vertex[0] <= edge_value
        elif edge_type == "bottom":
            current_inside = current[1] >= edge_value
            next_inside = next_vertex[1] >= edge_value
        elif edge_type == "top":
            current_inside = current[1] <= edge_value
            next_inside = next_vertex[1] <= edge_value
        else:
            continue

        if current_inside and next_inside:

            result.append(next_vertex)
        elif current_inside and not next_inside:

            intersection = compute_intersection(
                current, next_vertex, edge_type, edge_value
            )
            if intersection:
                result.append(intersection)
        elif not current_inside and next_inside:

            intersection = compute_intersection(
                current, next_vertex, edge_type, edge_value
            )
            if intersection:
                result.append(intersection)
            result.append(next_vertex)

    return result


def compute_intersection(p1, p2, edge_type, edge_value):
    x1, y1 = p1
    x2, y2 = p2

    if edge_type in ["left", "right"]:
        if abs(x2 - x1) < 1e-10:
            return None
        t = (edge_value - x1) / (x2 - x1)
        if 0 <= t <= 1:
            return (edge_value, y1 + t * (y2 - y1))
    else:
        if abs(y2 - y1) < 1e-10:
            return None
        t = (edge_value - y1) / (y2 - y1)
        if 0 <= t <= 1:
            return (x1 + t * (x2 - x1), edge_value)

    return None


def clip_polygon_to_screen(polygon, left, right, bottom, top):
    if not polygon or len(polygon) < 3:
        return []

    clipped = polygon

    clipped = clip_polygon_edge(clipped, "left", left)
    if not clipped:
        return []

    clipped = clip_polygon_edge(clipped, "right", right)
    if not clipped:
        return []

    clipped = clip_polygon_edge(clipped, "bottom", bottom)
    if not clipped:
        return []

    clipped = clip_polygon_edge(clipped, "top", top)

    return clipped if len(clipped) >= 3 else []


def clamp_color_component(value):
    return max(0.0, min(1.0, value))


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 characters long")
    try:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)
    except ValueError:
        raise ValueError("Invalid hex color format")


class LightSource:

    def __init__(
        self,
        pos: Vector3,
        color: str,
        brightness: float,
        falloff_type: str = "linear",
        falloff_rate: float = 0.1,
    ):
        self.pos = pos
        self.color = hex_to_rgb(color)
        self.color_hex = color
        self.brightness = max(0.0, brightness)
        self.falloff_type = falloff_type
        self.falloff_rate = max(0.001, falloff_rate)

    def calculate_falloff(self, distance: float) -> float:
        if self.falloff_type == "none":
            return 1.0
        elif self.falloff_type == "linear":
            return max(0.0, 1.0 - (distance * self.falloff_rate))
        elif self.falloff_type == "quadratic":
            return max(0.0, 1.0 - (distance * distance * self.falloff_rate))
        elif self.falloff_type == "exponential":
            return exp(-distance * self.falloff_rate)
        else:
            return max(0.0, 1.0 - (distance * self.falloff_rate))

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        return (0.0, 0.0, 0.0)


class AmbientLight(LightSource):

    def __init__(self, color: str, brightness: float):
        super().__init__(Vector3(0, 0, 0), color, brightness, "none", 0.0)

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        return (
            self.color[0] * self.brightness,
            self.color[1] * self.brightness,
            self.color[2] * self.brightness,
        )


class PointLight(LightSource):

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        light_dir = self.pos - point
        distance = light_dir.magnitude()

        if distance == 0:
            return (0.0, 0.0, 0.0)

        light_dir = light_dir.normalize()
        falloff = self.calculate_falloff(distance)
        dot_product = max(0.0, normal.dot(light_dir))
        intensity = self.brightness * falloff * dot_product

        return (
            self.color[0] * intensity,
            self.color[1] * intensity,
            self.color[2] * intensity,
        )


class DirectionalLight(LightSource):

    def __init__(
        self,
        pos: Vector3,
        direction: Vector3,
        color: str,
        brightness: float,
        falloff_type: str = "linear",
        falloff_rate: float = 0.0,
    ):
        super().__init__(pos, color, brightness, falloff_type, falloff_rate)
        self.direction = direction.normalize()

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        light_dir = -self.direction
        distance = (point - self.pos).magnitude()
        falloff = self.calculate_falloff(distance)
        dot_product = max(0.0, normal.dot(light_dir))
        intensity = self.brightness * falloff * dot_product

        return (
            self.color[0] * intensity,
            self.color[1] * intensity,
            self.color[2] * intensity,
        )


class SpotLight(LightSource):

    def __init__(
        self,
        pos: Vector3,
        direction: Vector3,
        color: str,
        brightness: float,
        cone_angle: float = radians(30),
        falloff_type: str = "linear",
        falloff_rate: float = 0.1,
    ):
        super().__init__(pos, color, brightness, falloff_type, falloff_rate)
        self.direction = direction.normalize()
        self.cone_angle = cone_angle

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        light_to_point = point - self.pos
        distance = light_to_point.magnitude()

        if distance == 0:
            return (0.0, 0.0, 0.0)

        light_to_point_normalized = light_to_point.normalize()
        dot_product_direction = self.direction.dot(light_to_point_normalized)
        dot_product_direction = max(-1.0, min(1.0, dot_product_direction))
        angle_to_point = acos(dot_product_direction)

        if angle_to_point > self.cone_angle:
            return (0.0, 0.0, 0.0)

        spotlight_factor = max(0.0, cos(angle_to_point))
        light_dir = -light_to_point_normalized
        distance_falloff = self.calculate_falloff(distance)
        dot_product = max(0.0, normal.dot(light_dir))
        intensity = self.brightness * distance_falloff * dot_product * spotlight_factor

        return (
            self.color[0] * intensity,
            self.color[1] * intensity,
            self.color[2] * intensity,
        )


class EmissiveLight(LightSource):

    def calculate_lighting(
        self, point: Vector3, normal: Vector3
    ) -> Tuple[float, float, float]:
        light_dir = self.pos - point
        distance = light_dir.magnitude()

        if distance == 0:
            return (0.0, 0.0, 0.0)

        light_dir = light_dir.normalize()
        falloff = self.calculate_falloff(distance)
        dot_product = max(0.0, normal.dot(light_dir))
        intensity = self.brightness * falloff * dot_product

        return (
            self.color[0] * intensity,
            self.color[1] * intensity,
            self.color[2] * intensity,
        )


class LightingConfig:
    __slots__ = [
        "light_sources",
        "use_caching",
        "enable_shadows",
        "light_bounces",
        "light_bounce_samples",
        "precompute_bounces",
        "use_advanced_lighting",
        "max_light_distance",
        "max_bounce_distance",
        "shadow_bias",
        "light_contribution_threshold",
        "_lighting_cache",
        "_cache_frame",
        "_temp_light",
        "_bounce_cache",
        "_bounce_cache_valid",
        "_face_lighting_cache",
        "_centroid_normal_cache",
        "_cache_hits",
        "_cache_misses",
        "_precompute_objects",
        "_max_distance_sq",
        "_max_bounce_distance_sq",
        "_precomputed_lighting_cache",
        "_precomputed_lighting_valid",
        "_simple_lighting_result",
    ]

    def __init__(
        self,
        use_advanced_lighting=True,
        max_light_distance=50.0,
        use_caching=True,
        light_bounces=0,
        light_bounce_samples=8,
        precompute_bounces=False,
        max_bounce_distance=8.0,
        light_contribution_threshold=0.001,
        enable_shadows=False,
        shadow_bias=0.001,
    ):
        self.light_sources = []
        self.use_caching = use_caching
        self.enable_shadows = enable_shadows
        self.light_bounces = max(0, light_bounces)
        self.light_bounce_samples = max(1, light_bounce_samples)
        self.precompute_bounces = precompute_bounces
        self.use_advanced_lighting = use_advanced_lighting
        self.max_light_distance = max_light_distance
        self.max_bounce_distance = max_bounce_distance
        self.shadow_bias = shadow_bias
        self.light_contribution_threshold = light_contribution_threshold
        self._lighting_cache = {}
        self._cache_frame = 0
        self._temp_light = [0.0, 0.0, 0.0]
        self._bounce_cache = {}
        self._bounce_cache_valid = False
        self._face_lighting_cache = {}
        self._centroid_normal_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._precompute_objects = None
        self._max_distance_sq = max_light_distance * max_light_distance
        self._max_bounce_distance_sq = max_bounce_distance * max_bounce_distance
        self._precomputed_lighting_cache = {}
        self._precomputed_lighting_valid = False
        self._simple_lighting_result = (1.0, 1.0, 1.0)

    def add_light_source(self, light_source):
        self.light_sources.append(light_source)
        if self.use_caching and len(self._lighting_cache) > 1000:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()
            self._centroid_normal_cache.clear()
        self._bounce_cache_valid = False
        self._precomputed_lighting_valid = False

    def remove_light_source(self, light_source):
        if light_source in self.light_sources:
            self.light_sources.remove(light_source)
            if self.use_caching:
                self._lighting_cache.clear()
                self._face_lighting_cache.clear()
                self._centroid_normal_cache.clear()
            self._bounce_cache_valid = False
            self._precomputed_lighting_valid = False

    def clear_light_sources(self):
        self.light_sources.clear()
        if self.use_caching:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()
            self._centroid_normal_cache.clear()
        self._bounce_cache_valid = False
        self._precomputed_lighting_valid = False

    def set_light_bounce_samples(self, samples):
        self.light_bounce_samples = max(1, samples)
        if self.use_caching:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()
        self._bounce_cache_valid = False

    def set_caching(self, enabled):
        self.use_caching = enabled
        if not enabled:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()
            self._centroid_normal_cache.clear()

    def set_shadows(self, enabled):
        self.enable_shadows = enabled
        if self.use_caching:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()

    def set_advanced_lighting(self, enabled):
        self.use_advanced_lighting = enabled
        if self.use_caching:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()

    def set_light_bounces(self, bounces):
        self.light_bounces = max(0, bounces)
        if self.use_caching:
            self._lighting_cache.clear()
            self._face_lighting_cache.clear()
        self._bounce_cache_valid = False

    def set_precompute_bounces(self, enabled):
        self.precompute_bounces = enabled
        if enabled:
            self._bounce_cache_valid = False
        else:
            self._bounce_cache.clear()

    def set_precompute_objects(self, objects):
        self._precompute_objects = objects
        self._bounce_cache_valid = False

    def _make_cache_key(self, point, normal):
        return (
            round(point.x * 200),
            round(point.y * 200),
            round(point.z * 200),
            round(normal.x * 200),
            round(normal.y * 200),
            round(normal.z * 200),
        )

    def _get_cached_centroid_normal(self, obj, face_idx):
        if not self.use_caching:
            return obj._compute_face_data(face_idx)

        cache_key = (id(obj), face_idx, obj._object_version)
        cached = self._centroid_normal_cache.get(cache_key)
        if cached is not None:
            return cached

        centroid, normal = obj._compute_face_data(face_idx)
        if len(self._centroid_normal_cache) < 2000:
            self._centroid_normal_cache[cache_key] = (centroid, normal)
        return centroid, normal

    def _get_cached_face_lighting(self, obj, face_idx, centroid, normal):
        if not self.use_caching:
            return self._compute_face_lighting(centroid, normal)

        cache_key = (
            id(obj),
            face_idx,
            obj._object_version,
            *self._make_cache_key(centroid, Vector3(0, 0, 0))[:3],
        )

        cached = self._face_lighting_cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1
        face_lighting = self._compute_face_lighting(centroid, normal)
        if len(self._face_lighting_cache) < 2000:
            self._face_lighting_cache[cache_key] = face_lighting
        return face_lighting

    def _compute_face_lighting(self, centroid, normal):
        face_lighting = [0.0, 0.0, 0.0]
        for light_source in self.light_sources:
            if isinstance(light_source, AmbientLight):
                ambient_contrib = light_source.calculate_lighting(centroid, normal)
                face_lighting[0] += ambient_contrib[0]
                face_lighting[1] += ambient_contrib[1]
                face_lighting[2] += ambient_contrib[2]
            else:
                dx = light_source.pos.x - centroid.x
                dy = light_source.pos.y - centroid.y
                dz = light_source.pos.z - centroid.z
                light_distance_sq = dx * dx + dy * dy + dz * dz

                if light_distance_sq > self._max_distance_sq:
                    continue

                light_contribution = light_source.calculate_lighting(centroid, normal)

                if (
                    light_contribution[0]
                    + light_contribution[1]
                    + light_contribution[2]
                ) < self.light_contribution_threshold:
                    continue

                face_lighting[0] += light_contribution[0]
                face_lighting[1] += light_contribution[1]
                face_lighting[2] += light_contribution[2]
        return face_lighting

    def precompute_bounce_lighting(self, objects=None):
        if not self.precompute_bounces or self.light_bounces == 0:
            return

        if objects is None:
            objects = self._precompute_objects

        if objects is None:
            return

        self._bounce_cache.clear()

        if isinstance(objects, Object):
            objects = [objects]

        flat_objects = []
        for obj_list in objects:
            if isinstance(obj_list, list):
                flat_objects.extend(obj_list)
            else:
                flat_objects.append(obj_list)
        objects = flat_objects

        processed = 0
        max_processed = 500

        for obj in objects:
            for face_idx in range(len(obj.faces)):
                if processed >= max_processed:
                    break

                centroid, normal = self._get_cached_centroid_normal(obj, face_idx)

                bounce_strength = 0.25
                total_bounce_lighting = [0.0, 0.0, 0.0]

                for bounce in range(min(self.light_bounces, 2)):
                    bounce_strength *= 0.8

                    if bounce_strength < 0.02:
                        break

                    bounce_lighting = self._calculate_bounce_lighting_optimized(
                        centroid,
                        normal,
                        objects,
                        obj,
                        face_idx,
                        max_distance=self.max_bounce_distance,
                        bounce_strength=bounce_strength,
                    )

                    bounce_total = (
                        bounce_lighting[0] + bounce_lighting[1] + bounce_lighting[2]
                    )
                    if bounce_total < self.light_contribution_threshold:
                        break

                    total_bounce_lighting[0] += bounce_lighting[0]
                    total_bounce_lighting[1] += bounce_lighting[1]
                    total_bounce_lighting[2] += bounce_lighting[2]

                cache_key = (
                    round(centroid.x * 100),
                    round(centroid.y * 100),
                    round(centroid.z * 100),
                    round(normal.x * 100),
                    round(normal.y * 100),
                    round(normal.z * 100),
                )

                self._bounce_cache[cache_key] = total_bounce_lighting
                processed += 1
            if processed >= max_processed:
                break

        self._bounce_cache_valid = True

    def precompute_direct_lighting(self, objects=None):
        if objects is None:
            objects = self._precompute_objects

        if objects is None:
            return

        self._precomputed_lighting_cache.clear()

        if isinstance(objects, Object):
            objects = [objects]

        flat_objects = []
        for obj_list in objects:
            if isinstance(obj_list, list):
                flat_objects.extend(obj_list)
            else:
                flat_objects.append(obj_list)
        objects = flat_objects

        processed = 0
        max_processed = 1000

        for obj in objects:
            for face_idx in range(len(obj.faces)):
                if processed >= max_processed:
                    break

                centroid, normal = self._get_cached_centroid_normal(obj, face_idx)

                lighting = self._compute_lighting(
                    centroid, normal, objects, obj, face_idx
                )

                cache_key = self._make_cache_key(centroid, normal)
                self._precomputed_lighting_cache[cache_key] = lighting
                processed += 1

            if processed >= max_processed:
                break

        self._precomputed_lighting_valid = True

    def _calculate_bounce_lighting_optimized(
        self,
        point,
        normal,
        objects,
        current_object,
        current_face_idx,
        max_distance=10.0,
        bounce_strength=0.5,
    ):
        bounce_lighting = [0.0, 0.0, 0.0]

        if bounce_strength < 0.02:
            return bounce_lighting

        num_samples = self.light_bounce_samples
        max_faces_per_object = 8
        sample_weight = 1.0 / num_samples

        up = (
            Vector3(0, 1, 0)
            if abs(normal.dot(Vector3(0, 1, 0))) <= 0.9
            else Vector3(1, 0, 0)
        )
        tangent = normal.cross(up).normalize()
        bitangent = normal.cross(tangent).normalize()

        two_pi = 2 * pi
        offset_amount = 0.4

        for i in range(num_samples):
            angle = (i / num_samples) * two_pi
            cos_angle = cos(angle)
            sin_angle = sin(angle)

            offset_dir_x = (
                tangent.x * cos_angle * offset_amount
                + bitangent.x * sin_angle * offset_amount
            )
            offset_dir_y = (
                tangent.y * cos_angle * offset_amount
                + bitangent.y * sin_angle * offset_amount
            )
            offset_dir_z = (
                tangent.z * cos_angle * offset_amount
                + bitangent.z * sin_angle * offset_amount
            )

            sample_dir = Vector3(
                normal.x + offset_dir_x,
                normal.y + offset_dir_y,
                normal.z + offset_dir_z,
            ).normalize()

            for obj in objects:
                if obj is current_object:
                    continue

                faces_checked = 0
                face_count = len(obj.faces)
                step = max(1, face_count // max_faces_per_object)

                for face_idx in range(0, face_count, step):
                    if faces_checked >= max_faces_per_object:
                        break
                    faces_checked += 1

                    face_center, face_normal = self._get_cached_centroid_normal(
                        obj, face_idx
                    )

                    dx = face_center.x - point.x
                    dy = face_center.y - point.y
                    dz = face_center.z - point.z
                    distance_sq = dx * dx + dy * dy + dz * dz

                    if distance_sq > self._max_bounce_distance_sq or distance_sq < 0.01:
                        continue

                    distance = sqrt(distance_sq)
                    inv_distance = 1.0 / distance
                    direction_to_face = Vector3(
                        dx * inv_distance, dy * inv_distance, dz * inv_distance
                    )

                    if normal.dot(direction_to_face) < 0.1:
                        continue
                    if (
                        face_normal.dot(
                            Vector3(
                                -direction_to_face.x,
                                -direction_to_face.y,
                                -direction_to_face.z,
                            )
                        )
                        < 0.1
                    ):
                        continue

                    sample_dot = sample_dir.dot(direction_to_face)
                    if sample_dot < 0.6:
                        continue

                    face_lighting = self._get_cached_face_lighting(
                        obj, face_idx, face_center, face_normal
                    )

                    face_lighting_total = (
                        face_lighting[0] + face_lighting[1] + face_lighting[2]
                    )
                    if face_lighting_total < self.light_contribution_threshold:
                        continue

                    falloff = max(0.0, 1.0 - (distance / max_distance))
                    angle_factor = max(0.0, normal.dot(direction_to_face))
                    surface_angle_factor = max(
                        0.0,
                        face_normal.dot(
                            Vector3(
                                -direction_to_face.x,
                                -direction_to_face.y,
                                -direction_to_face.z,
                            )
                        ),
                    )

                    bounce_factor = (
                        bounce_strength
                        * falloff
                        * angle_factor
                        * surface_angle_factor
                        * 0.25
                    )

                    if bounce_factor < 0.02:
                        continue

                    bounce_lighting[0] += face_lighting[0] * bounce_factor
                    bounce_lighting[1] += face_lighting[1] * bounce_factor
                    bounce_lighting[2] += face_lighting[2] * bounce_factor

        bounce_lighting[0] *= sample_weight
        bounce_lighting[1] *= sample_weight
        bounce_lighting[2] *= sample_weight

        return (
            max(0.0, min(1.0, bounce_lighting[0])),
            max(0.0, min(1.0, bounce_lighting[1])),
            max(0.0, min(1.0, bounce_lighting[2])),
        )

    def _calculate_realtime_bounce_lighting(
        self,
        point,
        normal,
        objects,
        current_object,
        current_face_idx,
    ):
        bounce_lighting = [0.0, 0.0, 0.0]

        if self.light_bounces == 0:
            return bounce_lighting

        bounce_strength = 0.25
        max_distance = self.max_bounce_distance
        num_samples = self.light_bounce_samples
        max_faces_per_object = 8

        for bounce in range(self.light_bounces):
            bounce_strength *= 0.8
            if bounce_strength < 0.02:
                break

            sample_weight = 1.0 / num_samples
            up = (
                Vector3(0, 1, 0)
                if abs(normal.dot(Vector3(0, 1, 0))) <= 0.9
                else Vector3(1, 0, 0)
            )
            tangent = normal.cross(up).normalize()
            bitangent = normal.cross(tangent).normalize()

            two_pi = 2 * pi
            offset_amount = 0.4

            for i in range(num_samples):
                angle = (i / num_samples) * two_pi
                cos_angle = cos(angle)
                sin_angle = sin(angle)

                offset_dir_x = (
                    tangent.x * cos_angle * offset_amount
                    + bitangent.x * sin_angle * offset_amount
                )
                offset_dir_y = (
                    tangent.y * cos_angle * offset_amount
                    + bitangent.y * sin_angle * offset_amount
                )
                offset_dir_z = (
                    tangent.z * cos_angle * offset_amount
                    + bitangent.z * sin_angle * offset_amount
                )

                sample_dir = Vector3(
                    normal.x + offset_dir_x,
                    normal.y + offset_dir_y,
                    normal.z + offset_dir_z,
                ).normalize()

                for obj in objects:
                    if obj is current_object:
                        continue

                    faces_checked = 0
                    face_count = len(obj.faces)
                    step = max(1, face_count // max_faces_per_object)

                    for face_idx in range(0, face_count, step):
                        if faces_checked >= max_faces_per_object:
                            break
                        faces_checked += 1

                        face_center, face_normal = self._get_cached_centroid_normal(
                            obj, face_idx
                        )

                        dx = face_center.x - point.x
                        dy = face_center.y - point.y
                        dz = face_center.z - point.z
                        distance_sq = dx * dx + dy * dy + dz * dz

                        if (
                            distance_sq > self._max_bounce_distance_sq
                            or distance_sq < 0.01
                        ):
                            continue

                        distance = sqrt(distance_sq)
                        inv_distance = 1.0 / distance
                        direction_to_face = Vector3(
                            dx * inv_distance, dy * inv_distance, dz * inv_distance
                        )

                        if normal.dot(direction_to_face) < 0.1:
                            continue
                        if (
                            face_normal.dot(
                                Vector3(
                                    -direction_to_face.x,
                                    -direction_to_face.y,
                                    -direction_to_face.z,
                                )
                            )
                            < 0.1
                        ):
                            continue

                        sample_dot = sample_dir.dot(direction_to_face)
                        if sample_dot < 0.6:
                            continue

                        face_lighting = self._get_cached_face_lighting(
                            obj, face_idx, face_center, face_normal
                        )

                        face_lighting_total = (
                            face_lighting[0] + face_lighting[1] + face_lighting[2]
                        )
                        if face_lighting_total < self.light_contribution_threshold:
                            continue

                        falloff = max(0.0, 1.0 - (distance / max_distance))
                        angle_factor = max(0.0, normal.dot(direction_to_face))
                        surface_angle_factor = max(
                            0.0,
                            face_normal.dot(
                                Vector3(
                                    -direction_to_face.x,
                                    -direction_to_face.y,
                                    -direction_to_face.z,
                                )
                            ),
                        )

                        bounce_factor = (
                            bounce_strength
                            * falloff
                            * angle_factor
                            * surface_angle_factor
                            * 0.25
                        )

                        if bounce_factor < 0.02:
                            continue

                        bounce_lighting[0] += face_lighting[0] * bounce_factor
                        bounce_lighting[1] += face_lighting[1] * bounce_factor
                        bounce_lighting[2] += face_lighting[2] * bounce_factor

            bounce_lighting[0] *= sample_weight
            bounce_lighting[1] *= sample_weight
            bounce_lighting[2] *= sample_weight

        return (
            max(0.0, min(1.0, bounce_lighting[0])),
            max(0.0, min(1.0, bounce_lighting[1])),
            max(0.0, min(1.0, bounce_lighting[2])),
        )

    def get_precomputed_bounce_lighting(self, point, normal):
        if not self.precompute_bounces or not self._bounce_cache_valid:
            return [0.0, 0.0, 0.0]

        cache_key = (
            round(point.x * 100),
            round(point.y * 100),
            round(point.z * 100),
            round(normal.x * 100),
            round(normal.y * 100),
            round(normal.z * 100),
        )

        return self._bounce_cache.get(cache_key, [0.0, 0.0, 0.0])

    def get_precomputed_lighting(self, point, normal):
        if not self._precomputed_lighting_valid:
            return None

        cache_key = self._make_cache_key(point, normal)
        return self._precomputed_lighting_cache.get(cache_key)

    def calculate_lighting_at_point(
        self, point, normal, objects=None, current_object=None, current_face_idx=None
    ):
        if not self.use_advanced_lighting:
            return self._simple_lighting_result

        precomputed = self.get_precomputed_lighting(point, normal)
        if precomputed is not None:
            return precomputed

        if not self.use_caching:
            return self._compute_lighting(
                point, normal, objects, current_object, current_face_idx
            )

        cache_key = self._make_cache_key(point, normal)
        cached_result = self._lighting_cache.get(cache_key)
        if cached_result is not None:
            self._cache_hits += 1
            return cached_result

        self._cache_misses += 1
        result = self._compute_lighting(
            point, normal, objects, current_object, current_face_idx
        )
        if len(self._lighting_cache) < 2000:
            self._lighting_cache[cache_key] = result
        return result

    def _compute_lighting(
        self, point, normal, objects=None, current_object=None, current_face_idx=None
    ):
        self._temp_light[0] = 0.0
        self._temp_light[1] = 0.0
        self._temp_light[2] = 0.0

        for light_source in self.light_sources:
            if isinstance(light_source, AmbientLight):
                light_contribution = light_source.calculate_lighting(point, normal)
                self._temp_light[0] += light_contribution[0]
                self._temp_light[1] += light_contribution[1]
                self._temp_light[2] += light_contribution[2]
            else:
                dx = light_source.pos.x - point.x
                dy = light_source.pos.y - point.y
                dz = light_source.pos.z - point.z
                distance_sq = dx * dx + dy * dy + dz * dz

                if distance_sq > self._max_distance_sq:
                    continue

                if (
                    self.enable_shadows
                    and objects
                    and is_point_occluded(
                        light_source.pos,
                        point,
                        objects,
                        current_object,
                    )
                ):
                    continue

                light_contribution = light_source.calculate_lighting(point, normal)
                contribution_total = (
                    light_contribution[0]
                    + light_contribution[1]
                    + light_contribution[2]
                )

                if contribution_total < self.light_contribution_threshold:
                    continue

                self._temp_light[0] += light_contribution[0]
                self._temp_light[1] += light_contribution[1]
                self._temp_light[2] += light_contribution[2]

        if self.light_bounces > 0 and objects:
            if self.precompute_bounces and self._bounce_cache_valid:
                bounce_lighting = self.get_precomputed_bounce_lighting(point, normal)
                self._temp_light[0] += bounce_lighting[0]
                self._temp_light[1] += bounce_lighting[1]
                self._temp_light[2] += bounce_lighting[2]
            else:
                bounce_lighting = self._calculate_realtime_bounce_lighting(
                    point, normal, objects, current_object, current_face_idx
                )
                self._temp_light[0] += bounce_lighting[0]
                self._temp_light[1] += bounce_lighting[1]
                self._temp_light[2] += bounce_lighting[2]

        return (
            max(0.0, min(1.0, self._temp_light[0])),
            max(0.0, min(1.0, self._temp_light[1])),
            max(0.0, min(1.0, self._temp_light[2])),
        )

    def get_cache_stats(self):
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "lighting_cache_size": len(self._lighting_cache),
            "face_lighting_cache_size": len(self._face_lighting_cache),
            "centroid_normal_cache_size": len(self._centroid_normal_cache),
            "bounce_cache_size": len(self._bounce_cache),
            "bounce_cache_valid": self._bounce_cache_valid,
            "precomputed_cache_size": len(self._precomputed_lighting_cache),
            "precomputed_valid": self._precomputed_lighting_valid,
        }


def set_as_light_source(obj: Dict, brightness: float, color: str = "#ffffff"):
    obj["is_light_source"] = True
    obj["emission_brightness"] = max(0.0, brightness)
    obj["emission_color_hex"] = color

    r, g, b = hex_to_rgb(color)
    obj["emission_color"] = (r, g, b)


def update_emissive_lights(objects, lighting_config):
    if isinstance(objects, Object):
        objects = [objects]

    emissive_lights = [
        light
        for light in lighting_config.light_sources
        if isinstance(light, EmissiveLight)
    ]
    for light in emissive_lights:
        lighting_config.remove_light_source(light)

    light_count = 0
    max_emissive_lights = 20

    for obj in objects:
        if obj.is_light_source and light_count < max_emissive_lights:
            emission_brightness = obj.emission_brightness
            emission_color_hex = obj.emission_color_hex

            face_step = max(1, len(obj.faces) // 5)

            for face_idx in range(0, len(obj.faces), face_step):
                if light_count >= max_emissive_lights:
                    break

                face, material = obj.faces[face_idx]
                face_center = sum(face, Vector3(0, 0, 0)) / len(face)
                emissive_light = EmissiveLight(
                    pos=face_center,
                    color=emission_color_hex,
                    brightness=emission_brightness,
                    falloff_type="linear",
                    falloff_rate=0.08,
                )

                lighting_config.add_light_source(emissive_light)
                light_count += 1


class Camera:
    __slots__ = [
        "pos",
        "direction",
        "lighting_config",
        "use_caching",
        "canvas",
        "root",
        "_render_items",
        "_temp_projected",
        "_screen_half_w",
        "_screen_half_h",
        "_fov_factor",
        "_aspect_ratio",
        "_projection_cache",
        "_visibility_cache",
        "_cache_frame",
        "_max_render_items",
        "_screen_width",
        "_screen_height",
        "_is_stereo_eye",
        "_render_offset_x",
        "_render_width",
    ]

    def __init__(
        self,
        pos,
        direction=None,
        lighting_config=None,
        use_caching=True,
        screen_width=800,
        screen_height=600,
        is_stereo_eye=False,
        render_offset_x=0,
        render_width=None,
        canvas=None,
        root=None,
    ):
        self.pos = pos
        if direction is None:
            self.direction = Vector3(0, 0, 1)
        else:
            self.direction = direction.normalize()
        self.lighting_config = lighting_config or LightingConfig()
        self.use_caching = use_caching

        self._is_stereo_eye = is_stereo_eye
        self._render_offset_x = render_offset_x
        self._render_width = render_width or screen_width

        if not is_stereo_eye:
            self.root = tk.Tk()
            self.root.title("3D Renderer")
            self.root.configure(bg="#181818")
            self.root.geometry(f"{screen_width}x{screen_height}")
            self.root.resizable(False, False)

            self.canvas = tk.Canvas(
                self.root,
                width=screen_width,
                height=screen_height,
                bg="#181818",
                highlightthickness=0,
            )
            self.canvas.pack()
        else:
            self.root = root
            self.canvas = canvas

        self._screen_width = screen_width
        self._screen_height = screen_height

        self._render_items = []
        self._temp_projected = []
        self._screen_half_w = self._render_width / 2.0
        self._screen_half_h = screen_height / 2.0
        self._fov_factor = 1.0
        self._aspect_ratio = self._render_width / screen_height
        self._projection_cache = {}
        self._visibility_cache = {}
        self._cache_frame = 0
        self._max_render_items = 2000

    def set_lighting_config(self, lighting_config):
        self.lighting_config = lighting_config

    def set_caching(self, enabled):
        self.use_caching = enabled
        if not enabled:
            self._projection_cache.clear()
            self._visibility_cache.clear()

    def move_axis(self, pos):
        self.pos += pos
        if self.use_caching:
            self._projection_cache.clear()
            self._visibility_cache.clear()
        return self.pos

    def move(self, steps, horizontal_only=False):
        if horizontal_only:
            forward = Vector3(self.direction.x, 0, self.direction.z).normalize()
        else:
            forward = self.direction

        move = forward * steps
        self.pos += move
        if self.use_caching:
            self._projection_cache.clear()
            self._visibility_cache.clear()
        return self.pos

    def strafe(self, steps):
        up = Vector3(0, 1, 0)
        right = self.direction.cross(up).normalize()
        move = right * steps
        self.pos += move
        if self.use_caching:
            self._projection_cache.clear()
            self._visibility_cache.clear()
        return self.pos

    def move_relative(self, pos, horizontal_only=False):
        self.move(pos.z, horizontal_only)
        self.strafe(pos.x)
        self.pos.y -= pos.y

    def rotate(self, pitch_delta, yaw_delta):
        up = Vector3(0, 1, 0)

        if yaw_delta != 0:
            cos_yaw = cos(yaw_delta)
            sin_yaw = sin(yaw_delta)
            new_x = self.direction.x * cos_yaw - self.direction.z * sin_yaw
            new_z = self.direction.x * sin_yaw + self.direction.z * cos_yaw
            self.direction = Vector3(new_x, self.direction.y, new_z).normalize()

        if pitch_delta != 0:
            cos_pitch = cos(pitch_delta)
            sin_pitch = sin(pitch_delta)
            new_direction = self.direction * cos_pitch + up * sin_pitch
            self.direction = new_direction.normalize()

        if self.use_caching and (pitch_delta != 0 or yaw_delta != 0):
            self._projection_cache.clear()
            self._visibility_cache.clear()

    def get_view_direction(self):
        return self.direction

    def _get_cached_projection(self, point, screen_width, screen_height, fov):
        if not self.use_caching:
            return self.project_point(point, screen_width, screen_height, fov)

        cache_key = (
            int(point.x * 1000),
            int(point.y * 1000),
            int(point.z * 1000),
            int(self.pos.x * 1000),
            int(self.pos.y * 1000),
            int(self.pos.z * 1000),
            int(self.direction.x * 1000),
            int(self.direction.y * 1000),
            int(self.direction.z * 1000),
            screen_width,
            screen_height,
            fov,
        )

        cached = self._projection_cache.get(cache_key)
        if cached is not None:
            return cached

        result = self.project_point(point, screen_width, screen_height, fov)
        if len(self._projection_cache) < 5000:
            self._projection_cache[cache_key] = result
        return result

    def _get_cached_visibility(self, face_center, normal):
        if not self.use_caching:
            return self.is_face_visible(face_center, normal)

        cache_key = (
            int(face_center.x * 1000),
            int(face_center.y * 1000),
            int(face_center.z * 1000),
            int(normal.x * 1000),
            int(normal.y * 1000),
            int(normal.z * 1000),
            int(self.pos.x * 1000),
            int(self.pos.y * 1000),
            int(self.pos.z * 1000),
        )

        cached = self._visibility_cache.get(cache_key)
        if cached is not None:
            return cached

        result = self.is_face_visible(face_center, normal)
        if len(self._visibility_cache) < 3000:
            self._visibility_cache[cache_key] = result
        return result

    def project_point(self, point, screen_width=800, screen_height=600, fov=90):
        relative_x = point.x - self.pos.x
        relative_y = point.y - self.pos.y
        relative_z = point.z - self.pos.z

        forward = self.direction
        up = Vector3(0, 1, 0)

        right_x = forward.y * up.z - forward.z * up.y
        right_y = forward.z * up.x - forward.x * up.z
        right_z = forward.x * up.y - forward.y * up.x

        right_mag = sqrt(right_x * right_x + right_y * right_y + right_z * right_z)
        if right_mag == 0:
            return None
        right_x /= right_mag
        right_y /= right_mag
        right_z /= right_mag

        actual_up_x = right_y * forward.z - right_z * forward.y
        actual_up_y = right_z * forward.x - right_x * forward.z
        actual_up_z = right_x * forward.y - right_y * forward.x

        x = relative_x * right_x + relative_y * right_y + relative_z * right_z
        y = (
            relative_x * actual_up_x
            + relative_y * actual_up_y
            + relative_z * actual_up_z
        )
        z = relative_x * forward.x + relative_y * forward.y + relative_z * forward.z

        if z <= 0:
            return None

        fov_rad = radians(fov)
        f = 1.0 / tan(fov_rad * 0.5)

        screen_x = (x * f / z) * (screen_height / screen_width)
        screen_y = y * f / z

        screen_x = screen_x * (screen_width * 0.5) + screen_width * 0.5
        screen_y = screen_y * (screen_height * 0.5) + screen_height * 0.5

        tkinter_x = screen_x + self._render_offset_x
        tkinter_y = screen_height - screen_y

        return (tkinter_x, tkinter_y)

    def is_face_visible(self, face_center, normal):
        view_x = face_center.x - self.pos.x
        view_y = face_center.y - self.pos.y
        view_z = face_center.z - self.pos.z
        view_mag = sqrt(view_x * view_x + view_y * view_y + view_z * view_z)
        if view_mag == 0:
            return False
        view_x /= view_mag
        view_y /= view_mag
        view_z /= view_mag
        dot_product = normal.x * view_x + normal.y * view_y + normal.z * view_z
        return dot_product < 0

    def _rgb_to_hex(self, color):
        r = max(0, min(255, int(color[0] * 255)))
        g = max(0, min(255, int(color[1] * 255)))
        b = max(0, min(255, int(color[2] * 255)))
        return f"#{r:02x}{g:02x}{b:02x}"

    def render(
        self,
        objects,
        materials,
        screen_width=None,
        screen_height=None,
        fov=90,
        show_normals=False,
        cull_back_faces=True,
        draw_light_sources=False,
        clear_canvas=True,
    ):
        if clear_canvas and not self._is_stereo_eye:
            self.canvas.delete("all")

        if screen_width is None:
            screen_width = self._render_width
        if screen_height is None:
            screen_height = self._screen_height

        try:
            if isinstance(objects, Object):
                objects = [objects]
            if isinstance(materials, Material):
                materials = {materials.name: materials.color}
            elif isinstance(materials, list):
                material_dict = {}
                for mat_list in materials:
                    if isinstance(mat_list, list):
                        for mat in mat_list:
                            material_dict[mat.name] = mat.color
                    else:
                        material_dict[mat_list.name] = mat_list.color
                materials = material_dict

            flat_objects = []
            for obj_list in objects:
                if isinstance(obj_list, list):
                    flat_objects.extend(obj_list)
                else:
                    flat_objects.append(obj_list)
            objects = flat_objects

            update_emissive_lights(objects, self.lighting_config)

            if not self.lighting_config._precomputed_lighting_valid:
                if self.lighting_config._precompute_objects is None:
                    self.lighting_config.set_precompute_objects(objects)
                self.lighting_config.precompute_direct_lighting(objects)

            if (
                self.lighting_config.precompute_bounces
                and not self.lighting_config._bounce_cache_valid
                and self.lighting_config.light_bounces > 0
            ):
                if self.lighting_config._precompute_objects is None:
                    self.lighting_config.set_precompute_objects(objects)
                self.lighting_config.precompute_bounce_lighting(objects)

            self._render_items.clear()
            self._cache_frame += 1

            if self.use_caching and self._cache_frame % 100 == 0:
                if len(self._projection_cache) > 7000:
                    self._projection_cache.clear()
                if len(self._visibility_cache) > 4000:
                    self._visibility_cache.clear()

            visible_faces = 0
            max_light_distance_sq = (
                self.lighting_config.max_light_distance
                * self.lighting_config.max_light_distance
            )

            for object in objects:
                if (
                    not object.faces
                    or len(self._render_items) >= self._max_render_items
                ):
                    continue

                if object.faces:
                    object_distance_sq = (
                        (object.faces[0][0][0].x - self.pos.x) ** 2
                        + (object.faces[0][0][0].y - self.pos.y) ** 2
                        + (object.faces[0][0][0].z - self.pos.z) ** 2
                    )
                    if object_distance_sq > max_light_distance_sq * 4:
                        continue

                is_light_source = object.is_light_source
                emission_brightness = object.emission_brightness
                emission_color = object.emission_color

                for face_idx, (face, material) in enumerate(object.faces):
                    if len(self._render_items) >= self._max_render_items:
                        break

                    if material:
                        material = f"{object.name}.{material}"

                    centroid, normal = object.get_face_data(face_idx)

                    distance_sq = (
                        (centroid.x - self.pos.x) ** 2
                        + (centroid.y - self.pos.y) ** 2
                        + (centroid.z - self.pos.z) ** 2
                    )
                    if distance_sq > max_light_distance_sq:
                        continue

                    distance = sqrt(distance_sq)

                    if cull_back_faces and not self._get_cached_visibility(
                        centroid, normal
                    ):
                        continue

                    visible_faces += 1

                    self._temp_projected.clear()
                    all_visible = True
                    for point in face:
                        proj = self._get_cached_projection(
                            point, screen_width, screen_height, fov
                        )
                        if proj is None:
                            all_visible = False
                            break
                        self._temp_projected.append(proj)

                    if not all_visible:
                        continue

                    lighting = self.lighting_config.calculate_lighting_at_point(
                        centroid, normal, objects, object, face_idx
                    )

                    if material in materials:
                        base_color = materials[material]
                        color = [
                            max(0.0, min(1.0, base_color[0] * lighting[0])),
                            max(0.0, min(1.0, base_color[1] * lighting[1])),
                            max(0.0, min(1.0, base_color[2] * lighting[2])),
                        ]
                    else:
                        avg_light = (lighting[0] + lighting[1] + lighting[2]) / 3.0
                        color = [avg_light, avg_light, avg_light]

                    if is_light_source:
                        color[0] = min(
                            1.0, color[0] + emission_color[0] * emission_brightness
                        )
                        color[1] = min(
                            1.0, color[1] + emission_color[1] * emission_brightness
                        )
                        color[2] = min(
                            1.0, color[2] + emission_color[2] * emission_brightness
                        )

                    self._render_items.append(
                        {
                            "type": "face",
                            "projected": list(self._temp_projected),
                            "color": color,
                            "distance": distance,
                        }
                    )

                    if show_normals:
                        normal_end = centroid + normal
                        normal_start_proj = self._get_cached_projection(
                            centroid, screen_width, screen_height, fov
                        )
                        normal_end_proj = self._get_cached_projection(
                            normal_end, screen_width, screen_height, fov
                        )

                        if normal_start_proj and normal_end_proj:
                            self._render_items.append(
                                {
                                    "type": "normal",
                                    "start": normal_start_proj,
                                    "end": normal_end_proj,
                                    "distance": distance,
                                }
                            )

            if draw_light_sources:
                for light_source in self.lighting_config.light_sources:
                    if isinstance(light_source, AmbientLight):
                        continue

                    light_distance_sq = (
                        (light_source.pos.x - self.pos.x) ** 2
                        + (light_source.pos.y - self.pos.y) ** 2
                        + (light_source.pos.z - self.pos.z) ** 2
                    )
                    if light_distance_sq > max_light_distance_sq:
                        continue

                    light_proj = self._get_cached_projection(
                        light_source.pos, screen_width, screen_height, fov
                    )
                    if light_proj:
                        light_distance = sqrt(light_distance_sq)
                        self._render_items.append(
                            {
                                "type": "light_source",
                                "position": light_proj,
                                "color": [
                                    max(0.0, min(1.0, color * light_source.brightness))
                                    for color in light_source.color
                                ],
                                "distance": light_distance,
                            }
                        )

            self._render_items.sort(key=lambda x: x["distance"], reverse=True)

            for item in self._render_items:
                if item["type"] == "face":
                    projected = item["projected"]
                    color = item["color"]

                    if len(projected) >= 3:
                        left_bound = self._render_offset_x
                        right_bound = self._render_offset_x + self._render_width
                        bottom_bound = 0
                        top_bound = screen_height

                        clipped = clip_polygon_to_screen(
                            projected, left_bound, right_bound, bottom_bound, top_bound
                        )

                        if clipped and len(clipped) >= 3:
                            flat_points = [
                                coord for point in clipped for coord in point
                            ]
                            hex_color = self._rgb_to_hex(color)

                            self.canvas.create_polygon(
                                flat_points, fill=hex_color, outline=hex_color, width=1
                            )

                elif item["type"] == "normal":
                    start_x, start_y = item["start"]
                    end_x, end_y = item["end"]

                    if (
                        self._render_offset_x
                        <= start_x
                        <= self._render_offset_x + screen_width
                        and 0 <= start_y <= screen_height
                        and self._render_offset_x
                        <= end_x
                        <= self._render_offset_x + screen_width
                        and 0 <= end_y <= screen_height
                    ):

                        self.canvas.create_line(
                            start_x, start_y, end_x, end_y, fill="red", width=2
                        )

                elif item["type"] == "light_source":
                    x, y = item["position"]
                    radius = 3

                    if (
                        self._render_offset_x - radius
                        <= x
                        <= self._render_offset_x + screen_width + radius
                        and -radius <= y <= screen_height + radius
                    ):

                        hex_color = self._rgb_to_hex(item["color"])

                        self.canvas.create_oval(
                            x - radius,
                            y - radius,
                            x + radius,
                            y + radius,
                            fill=hex_color,
                            outline=hex_color,
                        )

        except Exception as e:
            print(f"Warning: Error rendering item: {str(e)}")

    def update(self):
        if not self._is_stereo_eye:
            self.root.update()

    def mainloop(self):
        if not self._is_stereo_eye:
            self.root.mainloop()


class Head:
    def __init__(
        self,
        pos,
        direction=None,
        eye_separation=0.065,
        fov=90,
        screen_width=800,
        screen_height=600,
        screen_dist=10,
        invert=False,
        static=False,
        lighting_config=None,
        use_caching=True,
    ):
        self.pos = pos
        if direction is None:
            self.direction = Vector3(0, 0, 1)
        else:
            self.direction = direction.normalize()

        self.eye_separation = eye_separation
        self.fov = fov
        self.screen_dist = screen_dist
        self.invert = invert
        self.lighting_config = lighting_config or LightingConfig()
        self.use_caching = use_caching

        self.root = tk.Tk()
        self.root.title("Stereoscopic 3D Renderer")
        self.root.configure(bg="#181818")
        total_width = screen_width * 2 + screen_dist
        self.root.geometry(f"{total_width}x{screen_height}")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            self.root,
            width=total_width,
            height=screen_height,
            bg="#181818",
            highlightthickness=0,
        )
        self.canvas.pack()

        self._screen_width = screen_width
        self._screen_height = screen_height
        self._eye_width = screen_width

        if static and using_pillow:
            static_filenames = generate_static_images(
                screen_width, screen_height, name=f"{screen_width}x{screen_height}.{{}}"
            )
            self.static = [
                ImageTk.PhotoImage(Image.open(filename))
                for filename in static_filenames
            ]
        else:
            self.static = None
        self.static_loop = 0

        left_offset = 0
        right_offset = screen_width + screen_dist

        if invert:
            left_offset, right_offset = right_offset, left_offset

        up = Vector3(0, 1, 0)
        right_vec = self.direction.cross(up).normalize()

        left_eye_pos = self.pos - right_vec * (eye_separation / 2)
        right_eye_pos = self.pos + right_vec * (eye_separation / 2)

        self.left_eye = Camera(
            pos=left_eye_pos,
            direction=self.direction,
            lighting_config=self.lighting_config,
            use_caching=use_caching,
            screen_width=screen_width,
            screen_height=screen_height,
            is_stereo_eye=True,
            render_offset_x=left_offset,
            render_width=self._eye_width,
            canvas=self.canvas,
            root=self.root,
        )

        self.right_eye = Camera(
            pos=right_eye_pos,
            direction=self.direction,
            lighting_config=self.lighting_config,
            use_caching=use_caching,
            screen_width=screen_width,
            screen_height=screen_height,
            is_stereo_eye=True,
            render_offset_x=right_offset,
            render_width=self._eye_width,
            canvas=self.canvas,
            root=self.root,
        )

    def _update_eye_positions(self):
        up = Vector3(0, 1, 0)
        right_vec = self.direction.cross(up).normalize()

        self.left_eye.pos = self.pos - right_vec * (self.eye_separation / 2)
        self.right_eye.pos = self.pos + right_vec * (self.eye_separation / 2)
        self.left_eye.direction = self.direction
        self.right_eye.direction = self.direction

    def _update_eye_offsets(self, pixel_separation):
        base_left_offset = 0
        base_right_offset = self._eye_width + pixel_separation

        if self.invert:
            left_offset, right_offset = base_right_offset, base_left_offset
        else:
            left_offset, right_offset = base_left_offset, base_right_offset

        self.left_eye._render_offset_x = left_offset
        self.right_eye._render_offset_x = right_offset

        self.left_eye._render_width = self._eye_width
        self.right_eye._render_width = self._eye_width

        total_width = self._eye_width * 2 + pixel_separation
        current_width = self.canvas.winfo_width()
        if current_width != total_width:
            self.canvas.config(width=total_width)
            self.root.geometry(f"{total_width}x{self._screen_height}")

    def set_lighting_config(self, lighting_config):
        self.lighting_config = lighting_config
        self.left_eye.set_lighting_config(lighting_config)
        self.right_eye.set_lighting_config(lighting_config)

    def set_caching(self, enabled):
        self.use_caching = enabled
        self.left_eye.set_caching(enabled)
        self.right_eye.set_caching(enabled)

    def move_axis(self, pos):
        self.pos += pos
        self._update_eye_positions()
        return self.pos

    def move(self, steps, horizontal_only=False):
        if horizontal_only:
            forward = Vector3(self.direction.x, 0, self.direction.z).normalize()
        else:
            forward = self.direction

        move = forward * steps
        self.pos += move
        self._update_eye_positions()
        return self.pos

    def strafe(self, steps):
        up = Vector3(0, 1, 0)
        right = self.direction.cross(up).normalize()
        move = right * steps
        self.pos += move
        self._update_eye_positions()
        return self.pos

    def move_relative(self, pos, horizontal_only=False):
        self.move(pos.z, horizontal_only)
        self.strafe(pos.x)
        self.pos.y -= pos.y

    def rotate(self, pitch_delta, yaw_delta):
        current_pitch = asin(max(-1, min(1, self.direction.y)))

        new_pitch = current_pitch + pitch_delta
        max_pitch = pi / 2 - 0.01
        new_pitch = max(-max_pitch, min(max_pitch, new_pitch))

        horizontal_length = sqrt(self.direction.x**2 + self.direction.z**2)
        if horizontal_length > 0:
            current_yaw = atan2(self.direction.x, self.direction.z)
        else:
            current_yaw = 0

        new_yaw = current_yaw - yaw_delta

        cos_pitch = cos(new_pitch)
        self.direction = Vector3(
            cos_pitch * sin(new_yaw), sin(new_pitch), cos_pitch * cos(new_yaw)
        )

        self._update_eye_positions()

    def get_view_direction(self):
        return self.direction

    def render(
        self,
        objects,
        materials,
        screen_width=None,
        screen_height=None,
        fov=None,
        show_normals=False,
        cull_back_faces=True,
        draw_light_sources=False,
        pixel_separation=None,
    ):
        if pixel_separation is not None:
            self._update_eye_offsets(pixel_separation)

        self.canvas.delete("all")

        if screen_width is None:
            screen_width = self._eye_width
        if screen_height is None:
            screen_height = self._screen_height
        if fov is None:
            fov = self.fov

        if self.static is not None:

            left_center_x = self.left_eye._render_offset_x + screen_width // 2
            left_center_y = screen_height // 2
            right_center_x = self.right_eye._render_offset_x + screen_width // 2
            right_center_y = screen_height // 2

            self.canvas.create_image(
                left_center_x,
                left_center_y,
                image=self.static[self.static_loop % len(self.static)],
            )
            self.canvas.create_image(
                right_center_x,
                right_center_y,
                image=self.static[self.static_loop % len(self.static)],
            )
            self.static_loop += 1

        self.left_eye.render(
            objects=objects,
            materials=materials,
            screen_width=screen_width,
            screen_height=screen_height,
            fov=fov,
            show_normals=show_normals,
            cull_back_faces=cull_back_faces,
            draw_light_sources=draw_light_sources,
            clear_canvas=False,
        )

        self.right_eye.render(
            objects=objects,
            materials=materials,
            screen_width=screen_width,
            screen_height=screen_height,
            fov=fov,
            show_normals=show_normals,
            cull_back_faces=cull_back_faces,
            draw_light_sources=draw_light_sources,
            clear_canvas=False,
        )

    def update(self):
        self.root.update()

    def mainloop(self):
        self.root.mainloop()


class Material:

    def __init__(
        self,
        name: str,
        color: List[float],
        specular: List[float] = None,
        shininess: float = 0.0,
    ):
        self.name = name
        self.color = color
        self.specular = specular or [0.0, 0.0, 0.0]
        self.shininess = max(0.0, min(1000.0, shininess))


class Object:
    __slots__ = [
        "name",
        "faces",
        "is_light_source",
        "emission_brightness",
        "emission_color_hex",
        "emission_color",
        "use_caching",
        "_face_cache",
        "_object_version",
    ]

    def __init__(self, name: str, faces: List = None, use_caching=True):
        self.name = name
        self.faces = faces or []
        self.is_light_source = False
        self.emission_brightness = 0.0
        self.emission_color_hex = "#ffffff"
        self.emission_color = (1.0, 1.0, 1.0)
        self.use_caching = use_caching
        self._face_cache = {}
        self._object_version = 0

    def set_caching(self, enabled: bool):
        self.use_caching = enabled
        if not enabled:
            self._face_cache.clear()

    def _invalidate_cache(self):
        if self.use_caching:
            self._object_version += 1
            self._face_cache.clear()

    def set_as_light_source(self, brightness: float, color: str = "#ffffff"):
        self.is_light_source = True
        self.emission_brightness = max(0.0, brightness)
        self.emission_color_hex = color
        self.emission_color = hex_to_rgb(color)
        self._invalidate_cache()

    def precompute_face_cache(self):
        if not self.use_caching:
            return

        self._face_cache.clear()
        for face_idx in range(len(self.faces)):
            cache_key = (face_idx, self._object_version)
            if cache_key not in self._face_cache:
                result = self._compute_face_data(face_idx)
                self._face_cache[cache_key] = result

    def rotate(self, anchor: Vector3, rotation_vector: Vector3):
        angle = sqrt(rotation_vector.x**2 + rotation_vector.y**2 + rotation_vector.z**2)

        if angle == 0:
            return

        inv_angle = 1.0 / angle
        axis = Vector3(
            rotation_vector.x * inv_angle,
            rotation_vector.y * inv_angle,
            rotation_vector.z * inv_angle,
        )

        for face_idx, (face, material) in enumerate(self.faces):
            rotated_face = [
                vertex.rotate_point_around_axis(anchor, axis, angle) for vertex in face
            ]
            self.faces[face_idx] = (rotated_face, material)

        self._invalidate_cache()

    def move(self, axis: Vector3):
        for face_idx, (face, material) in enumerate(self.faces):
            moved_face = [vertex + axis for vertex in face]
            self.faces[face_idx] = (moved_face, material)

        self._invalidate_cache()

    def get_face_data(self, face_idx):
        if not self.use_caching:
            return self._compute_face_data(face_idx)

        cache_key = (face_idx, self._object_version)
        cached = self._face_cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._compute_face_data(face_idx)
        self._face_cache[cache_key] = result
        return result

    def _compute_face_data(self, face_idx):
        face, _ = self.faces[face_idx]
        centroid = calculate_face_centroid(face)
        normal = calculate_face_normal(face)
        return (centroid, normal)


def load_obj(file_path, scale=0) -> List[Object]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"OBJ file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading OBJ file {file_path}: {str(e)}")

    objects = []
    vertexes = []
    curr_mtl = None

    for line_num, line in enumerate(lines, 1):
        try:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            command, *args = parts

            if command == "o" and args:
                name = args[0]
                objects.append(Object(name, []))
                default_object_created = True
            elif command == "v" and len(args) >= 3:
                if scale == 0:
                    vertexes.append(Vector3(*map(float, args[:3])))
                else:
                    vertexes.append(Vector3(*map(lambda x: float(x) * scale, args[:3])))
            elif command == "f" and args:

                if not objects:
                    objects.append(Object("default", []))
                    default_object_created = True

                face_vertices = []
                for vertex_data in args:
                    try:

                        vertex_index = int(vertex_data.split("/")[0])

                        if vertex_index < 0:
                            vertex_index = len(vertexes) + vertex_index + 1
                        face_vertices.append(vertexes[vertex_index - 1])
                    except (ValueError, IndexError):
                        continue

                if len(face_vertices) >= 3:
                    objects[-1].faces.append((face_vertices, curr_mtl))
            elif command == "usemtl" and args:
                curr_mtl = args[0]
        except Exception as e:
            print(f"Warning: Error parsing line {line_num} in {file_path}: {str(e)}")
            continue

    if not objects:
        objects.append(Object("default", []))

    return objects


def load_mtl(file_path, objects) -> List[Material]:
    if isinstance(objects, Object):
        objects = [objects]

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Warning: MTL file not found: {file_path}")
        return []
    except Exception as e:
        print(f"Warning: Error reading MTL file {file_path}: {str(e)}")
        return []

    materials = []
    curr_mtl = None
    curr_kd = [1.0, 1.0, 1.0]
    curr_ks = [0.0, 0.0, 0.0]
    curr_ns = 0.0

    for line_num, line in enumerate(lines, 1):
        try:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            command, *args = parts

            if command == "newmtl" and args:
                if curr_mtl:
                    for obj in objects:
                        materials.append(
                            Material(
                                f"{obj.name}.{curr_mtl}", curr_kd, curr_ks, curr_ns
                            )
                        )
                curr_mtl = args[0]
                curr_kd = [1.0, 1.0, 1.0]
                curr_ks = [0.0, 0.0, 0.0]
                curr_ns = 0.0
            elif command == "Kd" and curr_mtl and len(args) >= 3:
                curr_kd = [max(0.0, min(1.0, float(i))) for i in args[:3]]
            elif command == "Ks" and curr_mtl and len(args) >= 3:
                curr_ks = [max(0.0, min(1.0, float(i))) for i in args[:3]]
            elif command == "Ns" and curr_mtl and args:
                curr_ns = max(0.0, min(1000.0, float(args[0])))
        except Exception as e:
            print(f"Warning: Error parsing line {line_num} in {file_path}: {str(e)}")
            continue

    if curr_mtl:
        for obj in objects:
            materials.append(
                Material(f"{obj.name}.{curr_mtl}", curr_kd, curr_ks, curr_ns)
            )

    return materials


def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


def accurate_sleep(seconds: int | float):
    if seconds == 0:
        return
    elif seconds < 0.05:
        target = time.perf_counter() + seconds
        while time.perf_counter() < target:
            pass
    else:
        time.sleep(seconds)


total = []
frame = 0
sample_freq = 10
fps_data = []
frames = []


def normalize_framerate(target, renderer=None):
    last_frame_time = [time.time()]

    def decorator(func):
        def wrapped(*args, **kwargs):
            global frame, fps_data
            current_time = time.time()

            actual_deltatime = current_time - last_frame_time[0]
            last_frame_time[0] = current_time

            deltatime = min(actual_deltatime, 1.0 / 15.0)

            result = func(deltatime, *args, **kwargs)

            frame_time = time.time() - current_time
            uncapped_fps = 1 / frame_time if frame_time > 0 else float("inf")

            fps_data.append(uncapped_fps)
            if len(fps_data) > 10:
                fps_data.pop(0)

            if renderer is not None:
                text_info = []

                try:
                    import psutil
                    import os

                    process = psutil.Process(os.getpid())
                    text_info.append(
                        f"MEM: {process.memory_info().rss / (1024 * 1024):.2f}MB"
                    )
                except ImportError:
                    text_info.append("MEM: N/A")

                fps_avg = sum(fps_data) / len(fps_data) if fps_data else 0

                time_to_sleep = max(0, (1 / target) - frame_time)

                _target = time.perf_counter() + time_to_sleep
                while time.perf_counter() < _target:
                    pass

                total_frame_time = time.time() - current_time
                capped_fps = 1 / total_frame_time if total_frame_time > 0 else target

                if hasattr(renderer, "left_eye"):
                    text_info.append(f"Renderer: Head (Stereoscopic)")
                    text_info.append(f"Eye separation: {renderer.eye_separation:.3f}m")
                    text_info.append(
                        f"Eye resolution: {renderer._eye_width}x{renderer._screen_height}"
                    )
                    text_info.append(f"Screen distance: {renderer.screen_dist}px")
                    text_info.append(f"Inverted: {renderer.invert}")
                else:
                    text_info.append(f"Renderer: Camera (Mono)")
                    text_info.append(
                        f"Resolution: {renderer._screen_width}x{renderer._screen_height}"
                    )

                if hasattr(renderer, "lighting_config") and renderer.lighting_config:
                    lc = renderer.lighting_config
                    text_info.append(f"Lighting:")
                    text_info.append(f"  Sources: {len(lc.light_sources)}")
                    text_info.append(f"  Bounces: {lc.light_bounces}")
                    text_info.append(f"  Shadows: {lc.enable_shadows}")
                    text_info.append(f"  Advanced: {lc.use_advanced_lighting}")

                    cache_stats = lc.get_cache_stats()
                    if cache_stats["hits"] + cache_stats["misses"] > 0:
                        text_info.append(
                            f"  Cache hit rate: {cache_stats['hit_rate']:.1f}%"
                        )

                text_info.append(f"FPS Data:")
                text_info.append(f"    real: {capped_fps:.1f}")
                text_info.append(f"    avg (uncapped): {fps_avg:.1f}")
                text_info.append(f"    uncapped: {uncapped_fps:.1f} | target: {target}")

                text_info.append(
                    f"Deltatime: {deltatime:.6f} | real: {actual_deltatime:.6f}"
                )

                text_info.append(
                    f"Position: ({renderer.pos.x:.2f}, {renderer.pos.y:.2f}, {renderer.pos.z:.2f})"
                )
                text_info.append(
                    f"Direction: ({renderer.direction.x:.2f}, {renderer.direction.y:.2f}, {renderer.direction.z:.2f})"
                )

                canvas = None
                if hasattr(renderer, "canvas"):
                    canvas = renderer.canvas
                elif hasattr(renderer, "left_eye") and hasattr(
                    renderer.left_eye, "canvas"
                ):
                    canvas = renderer.left_eye.canvas

                if canvas and isinstance(canvas, tk.Canvas):
                    for i, text in enumerate(text_info):
                        _left = 10
                        _right = renderer._screen_width + renderer.screen_dist + 20
                        canvas.create_text(
                            _left if not renderer.invert else _right,
                            10 + i * 15,
                            anchor="nw",
                            text=text,
                            fill="white",
                            font=("Arial", 8),
                        )

                        canvas.create_text(
                            _right if not renderer.invert else _left,
                            10 + i * 15,
                            anchor="nw",
                            text=text,
                            fill="white",
                            font=("Arial", 8),
                        )

                    compass_left_x = _left + 50
                    compass_right_x = _right + 50
                    compass_y = 360
                    compass_radius = 50

                    for compass_x in [compass_left_x, compass_right_x]:
                        canvas.create_oval(
                            compass_x - compass_radius,
                            compass_y - compass_radius,
                            compass_x + compass_radius,
                            compass_y + compass_radius,
                            outline="white",
                            fill="",
                            width=2,
                        )

                        origin_dir = Vector3(0, 0, 0) - renderer.pos
                        if origin_dir.magnitude() > 0.001:
                            origin_dir = origin_dir.normalize()

                            facing_dot = renderer.direction.dot(origin_dir)

                            up = Vector3(0, 1, 0)
                            right = renderer.direction.cross(up).normalize()
                            actual_up = right.cross(renderer.direction).normalize()

                            right_component = origin_dir.dot(right)
                            up_component = origin_dir.dot(actual_up)

                            line_length = compass_radius * 0.8
                            end_x = compass_x + right_component * line_length
                            end_y = compass_y - up_component * line_length

                            color_factor = (facing_dot + 1) / 2

                            red = int(128 * (1 - color_factor))
                            green = int(128 + 127 * color_factor)
                            blue = int(128 * (1 - color_factor))

                            line_color = f"#{red:02x}{green:02x}{blue:02x}"

                            canvas.create_line(
                                compass_x,
                                compass_y,
                                end_x,
                                end_y,
                                fill=line_color,
                                width=3,
                            )

                            arrow_angle = atan2(-(up_component), right_component)
                            arrow_size = 8

                            arrow_x1 = end_x - arrow_size * cos(arrow_angle - 0.5)
                            arrow_y1 = end_y - arrow_size * sin(arrow_angle - 0.5)
                            arrow_x2 = end_x - arrow_size * cos(arrow_angle + 0.5)
                            arrow_y2 = end_y - arrow_size * sin(arrow_angle + 0.5)

                            canvas.create_line(
                                end_x,
                                end_y,
                                arrow_x1,
                                arrow_y1,
                                fill=line_color,
                                width=2,
                            )
                            canvas.create_line(
                                end_x,
                                end_y,
                                arrow_x2,
                                arrow_y2,
                                fill=line_color,
                                width=2,
                            )

            return result

        return wrapped

    return decorator


def handle_movement(
    root,
    deltatime,
    speed=6.0,
    sensitivity=0.01,
):
    global current_mouse_x, current_mouse_y, last_mouse_x, last_mouse_y

    camera_movement = zero3()
    camera_angle = zero2()

    if not hasattr(root, "_pressed_keys"):
        root._pressed_keys = set()

    def key_press(event):
        root._pressed_keys.add(event.keysym.lower())
        return "break"

    def key_release(event):
        root._pressed_keys.discard(event.keysym.lower())
        return "break"

    def mouse_motion(event):
        global current_mouse_x, current_mouse_y
        current_mouse_x = event.x_root
        current_mouse_y = event.y_root

    root.bind("<KeyPress>", key_press)
    root.bind("<KeyRelease>", key_release)
    root.bind("<Motion>", mouse_motion)

    root.focus_set()

    keys = root._pressed_keys
    frame_speed = speed * deltatime

    if "w" in keys:
        camera_movement.z += frame_speed
    if "s" in keys:
        camera_movement.z -= frame_speed
    if "a" in keys:
        camera_movement.x -= frame_speed
    if "d" in keys:
        camera_movement.x += frame_speed

    if "control_l" in keys or "control_r" in keys:
        camera_movement.y += frame_speed
    if "space" in keys:
        camera_movement.y -= frame_speed

    try:
        frame_sensitivity = sensitivity * deltatime * 60
        camera_angle.x -= (current_mouse_y - last_mouse_y) * frame_sensitivity
        camera_angle.y += (current_mouse_x - last_mouse_x) * frame_sensitivity
    except NameError:
        pass

    last_mouse_x = current_mouse_x
    last_mouse_y = current_mouse_y

    return camera_movement, camera_angle


if __name__ == "__main__":
    current_mouse_x = 400
    current_mouse_y = 300
    last_mouse_x = 400
    last_mouse_y = 300

    lighting_config = LightingConfig(
        use_advanced_lighting=True,
        max_light_distance=50.0,
        use_caching=False,
        light_bounces=1000,
        light_bounce_samples=5,
        precompute_bounces=False,
        max_bounce_distance=10.0,
        light_contribution_threshold=0.001,
        enable_shadows=True,
        shadow_bias=0.001,
    )

    lighting_config.add_light_source(AmbientLight(color="#FFFFFF", brightness=0.1))
    lighting_config.add_light_source(
        PointLight(
            pos=Vector3(-0.5, 1, -1.5),
            color="#008cff",
            brightness=2.0,
            falloff_type="linear",
            falloff_rate=0.04,
        )
    )
    lighting_config.add_light_source(
        PointLight(
            pos=Vector3(0.5, 1, -1.5),
            color="#04ff00",
            brightness=2.0,
            falloff_type="linear",
            falloff_rate=0.04,
        )
    )
    lighting_config.add_light_source(
        PointLight(
            pos=Vector3(-0.5, -1, -1.5),
            color="#ffb300",
            brightness=2.0,
            falloff_type="linear",
            falloff_rate=0.04,
        )
    )
    lighting_config.add_light_source(
        PointLight(
            pos=Vector3(0.5, -1, -1.5),
            color="#f2ff00",
            brightness=2.0,
            falloff_type="linear",
            falloff_rate=0.04,
        )
    )
    lighting_config.add_light_source(
        PointLight(
            pos=Vector3(0.5, 0, 2),
            color="#ff0000",
            brightness=2.0,
            falloff_type="linear",
            falloff_rate=0.04,
        )
    )
    lighting_config.add_light_source(
        PointLight(
            pos=Vector3(-0.5, 0, 2),
            color="#aa00ff",
            brightness=2.0,
            falloff_type="linear",
            falloff_rate=0.04,
        )
    )

    head = Head(
        pos=Vector3(0, 5, -15),
        direction=Vector3(0, -0.2, 1).normalize(),
        eye_separation=0.15,
        fov=90,
        screen_width=600,
        screen_height=600,
        screen_dist=20,
        invert=True,
        static=True,
        lighting_config=lighting_config,
        use_caching=False,
    )

    cube = load_obj(resource_path("objs/cube.obj"), scale=0.2)
    cube_mtl = load_mtl(resource_path("objs/cube.mtl"), cube)
    cube[0].move(Vector3(0, 0, -4))

    cubefs = load_obj(resource_path("objs/cube.obj"), scale=0.2)
    cubefs_mtl = load_mtl(resource_path("objs/cube.mtl"), cube)
    cubefs[0].move(Vector3(2, 0, -4))

    oes = load_obj(resource_path("objs/oes.obj"), scale=0.2)
    oes_mtl = load_mtl(resource_path("objs/oes.mtl"), cube)
    oes[0].move(Vector3(0, 0, -2))

    blahaj = load_obj(resource_path("objs/blahaj.obj"), scale=0.5)
    blahaj_mtl = load_mtl(resource_path("objs/blahaj.mtl"), blahaj)

    objects = cube + cubefs + blahaj + oes
    materials = cube_mtl + cubefs_mtl + blahaj_mtl + oes_mtl

    lighting_config.set_precompute_objects(objects)
    lighting_config.precompute_direct_lighting(objects)
    lighting_config.precompute_bounce_lighting(objects)

    for obj in objects:
        obj.precompute_face_cache()

    lighting_config.set_precompute_objects(objects)

    velocity = zero3()

    @normalize_framerate(60, head)
    def render_frame(deltatime):
        global velocity

        movement, rotation = handle_movement(
            head.root, deltatime, speed=2.0, sensitivity=0.01
        )

        velocity += movement
        velocity *= 0.8

        head.move_relative(velocity, horizontal_only=True)
        head.rotate(rotation.x, rotation.y)

        head.render(
            objects=objects,
            materials=materials,
            show_normals=False,
            cull_back_faces=True,
            draw_light_sources=True,
        )

    try:
        while True:
            render_frame()
            head.update()
    except KeyboardInterrupt:
        print("\nShutting down renderer...")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        try:
            head.root.quit()
        except:
            pass
