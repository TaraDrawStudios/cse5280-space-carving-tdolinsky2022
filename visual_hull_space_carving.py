import numpy as np
import matplotlib.pyplot as plt

# camera adjustment and projection functions
def look_at_camera(camera_pos, target=np.array([0.0, 0.0, 0.0])):
    camera_pos = np.array(camera_pos, dtype=float)
    target = np.array(target, dtype=float)

    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)

    up_guess = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_guess)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    R = np.vstack([right, up, forward])
    T = -R @ camera_pos
    return R, T

# projection of 3D points to 2D image plane
def project_points(points, K, R, T):
    X_cam = (R @ points.T).T + T

    z = X_cam[:, 2]
    visible = z > 1e-6

    x_norm = X_cam[:, 0] / z
    y_norm = X_cam[:, 1] / z

    u = K[0, 0] * x_norm + K[0, 2]
    v = K[1, 1] * y_norm + K[1, 2]

    pixels = np.vstack([u, v]).T
    return pixels, visible

# data structures and utilities for the main algorithm
def make_object_points():
    points = []

    rng = np.linspace(-0.8, 0.8, 50)
    for x in rng:
        for y in rng:
            for z in rng:
                r = np.sqrt(x*x + y*y + z*z)
                if 0.6 <= r <= 0.8:
                    points.append([x, y, z])

    return np.array(points)

# main functions for visual hull space carving
def create_silhouette(points, K, R, T, image_size):
    H, W = image_size
    pixels, visible = project_points(points, K, R, T)

    mask = np.zeros((H, W), dtype=bool)
    pix = pixels[visible]

    u = np.round(pix[:, 0]).astype(int)
    v = np.round(pix[:, 1]).astype(int)

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[inside]
    v = v[inside]

    mask[v, u] = True

    for _ in range(5):
        padded = np.pad(mask, 1)
        mask = (
            padded[1:-1, 1:-1] |
            padded[:-2, 1:-1] |
            padded[2:, 1:-1] |
            padded[1:-1, :-2] |
            padded[1:-1, 2:]
        )

    return mask

# main data loading and preparation function
def build_voxel_grid(resolution=40, bounds=(-1.2, 1.2)):
    xs = np.linspace(bounds[0], bounds[1], resolution)
    ys = np.linspace(bounds[0], bounds[1], resolution)
    zs = np.linspace(bounds[0], bounds[1], resolution)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    voxels = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return voxels, X.shape

# main space carving function
def space_carve(voxels, grid_shape, silhouettes, cameras, K, image_size):
    H, W = image_size
    occupied = np.ones(len(voxels), dtype=bool)

    for i, (mask, (R, T)) in enumerate(zip(silhouettes, cameras)):
        pixels, visible = project_points(voxels, K, R, T)

        u = np.round(pixels[:, 0]).astype(int)
        v = np.round(pixels[:, 1]).astype(int)

        inside_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid = visible & inside_img

        inside_silhouette = np.zeros(len(voxels), dtype=bool)
        idx = np.where(valid)[0]
        inside_silhouette[idx] = mask[v[idx], u[idx]]

        occupied &= inside_silhouette
        print(f"View {i+1}: {occupied.sum()} voxels left")

    return occupied.reshape(grid_shape)

# main function to load data and run the algorithm
def visualize(occupancy_grid):
    voxels = np.argwhere(occupancy_grid)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(voxels[:,0], voxels[:,1], voxels[:,2], s=2)

    plt.title("Visual Hull")
    plt.show()


def main():
    image_size = (256, 256)
    H, W = image_size

    fx = fy = 200
    cx = W / 2
    cy = H / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    object_points = make_object_points()

    cameras = []
    silhouettes = []

    num_views = 15
    radius = 3

    # create cameras and silhouettes
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views
        camera_pos = np.array([
            radius * np.cos(theta),
            0.5,
            radius * np.sin(theta)
        ])

        R, T = look_at_camera(camera_pos)
        cameras.append((R, T))

        mask = create_silhouette(object_points, K, R, T, image_size)
        silhouettes.append(mask)

    # create 
    voxels, grid_shape = build_voxel_grid()
    occupancy_grid = space_carve(voxels, grid_shape, silhouettes, cameras, K, image_size)

    visualize(occupancy_grid)


if __name__ == "__main__":
    main()