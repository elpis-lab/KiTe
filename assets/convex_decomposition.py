import trimesh
import coacd
import numpy as np
import os


def do_coacd(filename, **kwargs):
    """Run COACD"""
    mesh = trimesh.load(filename)
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(mesh, **kwargs)
    mesh_parts = []
    for vs, fs in parts:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    return mesh_parts


def main(folders, override=False, max_convex_hull=4, convex_face_count=32):
    """Run convex decomposition and save individual convex parts"""
    asset_dir = os.path.abspath(os.path.dirname(__file__))
    for folder in folders:
        mesh_path = os.path.join(asset_dir, folder)
        mesh_file = os.path.join(mesh_path, "textured.obj")
        test_out_file = os.path.join(mesh_path, f"textured_coacd_0.stl")
        if not override and os.path.exists(test_out_file):
            print(f"Skipping {folder} because it already exists")
            continue

        # Do convex decomposition
        mesh_parts = do_coacd(
            mesh_file,
            max_convex_hull=max_convex_hull,
            # max_ch_vertex=32,
            # decimate=True,
        )

        # Save the decomposed parts as separate STL files.
        for i, p in enumerate(mesh_parts):
            # simplify each convex hull
            p = p.simplify_quadric_decimation(face_count=convex_face_count)
            submesh_name = os.path.join(mesh_path, f"textured_coacd_{i}.stl")
            p.export(submesh_name)


if __name__ == "__main__":
    folders = ["banana", "cracker_box_flipped", "mustard_bottle_flipped"]
    main(folders, override=True)
