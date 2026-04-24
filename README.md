# Visual Hull Reconstruction via Space Carving

# Objective

The goal of this assignment was to reconstruct a 3D shape using multiple 2D views. This was done by generating silhouettes from different camera angles, projecting 3D points into image space using the pinhole camera model, and applying space carving to obtain the visual hull of the object.

# Method

### Multi-View Silhouettes

A synthetic 3D object was created using sampled points forming a spherical shape. Multiple virtual cameras were placed around the object in a circular arrangement. For each camera view, the object points were projected into image space to generate binary silhouettes.

### Pinhole Camera Model

Projection from 3D to 2D was implemented using the pinhole camera equation:

\[
x = K [R | T]
\]

Where:
- \(X\) represents a 3D point
- \(R\) and \(T\) are the rotation and translation matrices
- \(K\) is the intrinsic camera matrix
- \(x\) is the projected 2D point

A visibility condition was applied so that only points with positive depth (\(z > 0\)) were considered valid.

### Space Carving

A 3D voxel grid was initialized to enclose the object. Each voxel was projected into every camera view. A voxel was kept only if:
- It projected inside the image boundaries
-It had positive depth
- It landed inside the silhouette in all views

If a voxel failed any of these conditions, it was removed. This process was repeated across all views, gradually carving away inconsistent voxels.

### Visualization

The remaining voxels after carving were reshaped into a 3D occupancy grid. These voxels were visualized as a point cloud using a 3D scatter plot.

# Results

The number of voxels decreased as more views were processed, showing that inconsistent voxels were successfully removed. The process eventually stabilized, indicating convergence to a consistent visual hull.

The final reconstruction appears as a solid, rounded shape approximating the original object. The result is slightly blocky due to the discrete voxel grid but still captures the overall structure.

# Experiments

### Effect of Number of Views

Increasing the number of camera views improved the reconstruction quality. With fewer views, the visual hull was less constrained and appeared more approximate. With more views, the shape became more refined and accurate.

### Effect of Voxel Resolution

Higher voxel resolution resulted in a more detailed reconstruction but increased computation time. Lower resolution grids were faster but produced coarser results.

#Limitations

The visual hull method cannot reconstruct concave regions that are not visible in silhouettes. It produces the largest volume consistent with all views, which can lead to overestimation of the true shape.

Additionally, the reconstruction depends heavily on silhouette quality. Any inaccuracies in the silhouettes can affect the final result.

# Implementation Notes

PyTorch3D and Open3D were not used due to compatibility issues with on Windows. Several updates were made but none worked. Instead, silhouettes were generated using synthetic data, and visualization was performed using matplotlib.

Despite this adjustment, all core components of the assignment were implemented, including projection, silhouette consistency, and space carving.

# Conclusion

The assignment successfully demonstrated how multiple 2D views can be used to reconstruct a 3D shape. The space carving algorithm effectively removed inconsistent voxels, resulting in a visual hull that approximates the object. The experiment highlights both the strengths and limitations of silhouette-based reconstruction methods.
