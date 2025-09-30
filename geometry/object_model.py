import numpy as np
import trimesh
from .point_cloud import ObjectPointCloud


def get_obj_shape(obj_name):
    """Get the shape of the object"""
    mesh = trimesh.load(obj_name)
    shape = mesh.bounds[1] - mesh.bounds[0]
    return shape


def get_obj_2d_points(obj_name, n_points=200, slice_height=0.0):
    """Get the 2D points of the object"""
    pcd = ObjectPointCloud(obj_name, n_points, True, slice_height)
    return pcd.points[:, :2], pcd.normals[:, :2]
