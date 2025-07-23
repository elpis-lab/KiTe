import trimesh
import coacd
import numpy as np
import os


def do_coacd(filename, **kwargs):
    mesh = trimesh.load(filename)
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(mesh, **kwargs)
    mesh_parts = []
    for vs, fs in parts:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    return mesh_parts


def main(override=False):
    asset_dir = os.path.abspath(os.path.dirname(__file__))
    folders = ["banana", "cracker_box_flipped", "mustard_bottle_flipped"]
    max_convex_hull = 4
    max_ch_vertex = 32

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
            decimate=True,
            max_ch_vertex=max_ch_vertex,
        )

        # Save the decomposed parts as separate STL files.
        for i, p in enumerate(mesh_parts):
            submesh_name = os.path.join(mesh_path, f"textured_coacd_{i}.stl")
            p.export(submesh_name)


if __name__ == "__main__":
    main(override=False)
