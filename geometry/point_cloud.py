import open3d as o3d
import numpy as np
import trimesh
import rtree
from shapely.geometry import LineString
from shapely.geometry import Point as ShapelyPoint
from shapely.ops import unary_union


class ObjectPointCloud:
    def __init__(
        self,
        mesh_path,
        num_points=500,
        slice_z=False,
        slice_height=0.0,
    ):
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.mesh.compute_vertex_normals()

        # Use all 3d Point Cloud
        if not slice_z:
            # Sample in object frame
            pcd = self.mesh.sample_points_uniformly(
                number_of_points=num_points
            )
            self.points = np.asarray(pcd.points)
            self.normals = np.asarray(pcd.normals)

        # Use only a slice of the point cloud
        else:
            self.points, self.normals = self.build_from_slice_2d(
                mesh_path, num_points, slice_height
            )

    def build_from_slice_2d(
        self, mesh_path: str, num_points: int, slice_height: float
    ):
        # Use trimesh for sectioning
        tri = trimesh.load(mesh_path, force="mesh")
        if not isinstance(tri, trimesh.Trimesh):
            if isinstance(tri, trimesh.Scene):
                tri = trimesh.util.concatenate(tri.dump())
            else:
                raise ValueError("Unsupported mesh file for slicing.")

        # Compute cross-section with plane
        section = tri.section(
            plane_origin=(0.0, 0.0, slice_height), plane_normal=(0.0, 0.0, 1.0)
        )
        if section is None:
            raise ValueError(
                f"No intersection at z={slice_height}. Try another height."
            )

        # Turn the 3D cross-section path(s) into 2D coordinates (XY)
        planar, _ = section.to_planar()
        # "planar" is a Path2D with entities (polylines)
        # Consolidate into shapely polygons or polylines
        shapely_geoms = planar.polygons_full
        if not shapely_geoms:
            # Fall back to polylines if polygons are not formed (open contours)
            # We'll treat them as closed by linking endpoints if needed.
            # Convert each entity to LineString, then buffer to get area.
            ls = [e.discrete(200) for e in planar.entities]
            polys = [ShapelyPoint(0, 0).buffer(0)]  # dummy to start union
            for arr in ls:
                # attempt a small buffer around the polyline to create area
                poly = LineString(arr).buffer(1e-5)
                polys.append(poly)
            union_poly = unary_union(polys)
            if union_poly.is_empty:
                raise ValueError(
                    "Cross-section found but couldn't form polygons."
                )
            shapely_geoms = [union_poly]

        # Merge multiple polygons if needed;
        # choose the largest by area as the object cross-section
        shapely_geoms = [g for g in shapely_geoms if not g.is_empty]
        geom = max(shapely_geoms, key=lambda g: g.area)

        # Get an exterior boundary sample (Nx2)
        boundary = np.array(geom.exterior.coords)  # closed ring
        boundary = _resample_polyline(
            boundary, target_pts=max(2000, num_points * 5)
        )

        # Get centroid (for inward normal orientation)
        cx, cy = geom.centroid.x, geom.centroid.y
        center = np.array([cx, cy])

        # points2d = _sample_per_degree(
        #     boundary, center=center, total=num_points
        # )
        points2d = _sample_uniform_perimeter(boundary, num_points)
        # Compute 2D inward normals on the curve (use finite differences)
        normals2d = _normals_2d(points2d, center)

        # Save as 3D (at 0)
        points = np.c_[points2d, np.full(len(points2d), 0)]
        normals = np.c_[normals2d, np.zeros(len(points2d))]
        return points, normals

    def transform(self, matrix: np.ndarray):
        """Apply rigid transform to cached point cloud and rotate normals."""
        # Transform the point cloud
        points_h = np.hstack([self.points, np.ones((self.points.shape[0], 1))])
        transformed_points = (matrix @ points_h.T).T[:, :3]
        # Rotate the normals as well
        transformed_normals = (matrix[:3, :3] @ self.normals.T).T
        return transformed_points, transformed_normals

    def show(self):
        """Show the point cloud"""
        # Create a mesh from the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.normals = o3d.utility.Vector3dVector(self.normals)
        pcd.paint_uniform_color([1.0, 1.0, 1.0])

        # Build a LineSet for normals (each as a segment p -> p + scale*n)
        normal_scale = 0.01
        starts = self.points
        ends = self.points - normal_scale * self.normals
        verts = np.vstack([starts, ends])
        lines = np.array([[i, i + len(starts)] for i in range(len(starts))])
        n_lines = o3d.geometry.LineSet()
        n_lines.points = o3d.utility.Vector3dVector(verts)
        n_lines.lines = o3d.utility.Vector2iVector(lines)

        # Check size
        aabb = pcd.get_axis_aligned_bounding_box()
        dx, dy, dz = aabb.get_extent()
        print(f"Point Cloud AABB extent: {dx:.4f}, {dy:.4f}, {dz:.4f}")
        # Visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.add_geometry(n_lines)
        vis.get_render_option().point_size = 20
        vis.run()
        vis.destroy_window()


def _resample_polyline(poly: np.ndarray, target_pts: int) -> np.ndarray:
    """
    Resample a closed polyline to 'target_pts' approximately evenly spaced points.
    poly: (M, 2), first point == last point (closed)
    """
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    segs = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    L = segs.sum()
    if L == 0:
        return np.repeat(poly[:1], target_pts, axis=0)
    d = np.hstack([[0], np.cumsum(segs)])
    t = np.linspace(0, L, target_pts, endpoint=False)

    # interpolate along cumulative distance
    res = []
    j = 0
    for ti in t:
        while j + 1 < len(d) and d[j + 1] < ti:
            j += 1
        # interpolate between poly[j] and poly[j+1]
        seg_len = d[j + 1] - d[j]
        if seg_len == 0:
            res.append(poly[j])
        else:
            alpha = (ti - d[j]) / seg_len
            res.append((1 - alpha) * poly[j] + alpha * poly[j + 1])
    return np.array(res)


def _sample_uniform_perimeter(boundary: np.ndarray, total: int) -> np.ndarray:
    """
    Sample 'total' points uniformly along a (resampled) closed boundary polyline.
    """
    # boundary assumed dense; just pick indices uniformly
    idx = np.linspace(0, len(boundary) - 1, total, endpoint=False).astype(int)
    return boundary[idx]


def _safe_unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n < eps, eps, n)
    return v / n


def _normals_2d(points2d: np.ndarray, center: np.ndarray) -> np.ndarray:
    n = len(points2d)
    if n <= 1:
        return np.zeros((n, 2))

    prev = np.roll(points2d, 1, axis=0)
    nxt = np.roll(points2d, -1, axis=0)
    tang = _safe_unit(nxt - prev)
    normals = -_safe_unit(np.stack([-tang[:, 1], tang[:, 0]], axis=1))

    # If polygon is CW, flip to match outward for CCW convention
    x, y = points2d[:, 0], points2d[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    if signed_area < 0:
        normals *= -1.0
    return normals


if __name__ == "__main__":
    for obj in [
        "cracker_box_flipped",
        "mustard_bottle_flipped",
        "banana",
        "letter_t",
        "master_chef_can_flipped",
        "school_bus",
        "trash_truck",
    ]:
        obj = ObjectPointCloud(
            f"assets/{obj}/textured.obj",
            num_points=100,
            slice_z=True,
            slice_height=0.0,
        )
        # obj = ObjectPointCloud("assets/letter_t/textured.obj", num_points=180)
        obj.show()
