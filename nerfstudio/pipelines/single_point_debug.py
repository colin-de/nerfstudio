import torch


# This file is used for debugging and testing the single point pipeline
def debug_point(depths, cam_0_to_world, cam_1_to_world):
    intrinsics_test = torch.tensor(
        [[702.0630, 0.0000, 566.0000], [0.0000, 701.9382, 437.0000], [0.0000, 0.0000, 1.0000]], device="cuda:0"
    ).double()
    # depth_value = torch.tensor(depths[0][360, 700], device="cuda:0", dtype=torch.float64).double()
    depth_value = depths[0][360, 700].clone().detach().double()
    # Step 1: Create homogeneous coordinate
    homogeneous_pixel_coordinates = torch.tensor([700, 360, 1], dtype=torch.float64, device="cuda:0").double()
    # Step 2: Calculate the inverse of the intrinsic matrix
    inv_intrinsics = torch.inverse(intrinsics_test.double())
    # Step 3: Multiply by the inverse of the intrinsic and by the depth
    cam_0_coordinates = torch.matmul(homogeneous_pixel_coordinates, inv_intrinsics.T) * depth_value
    # Step 4: Convert to homogeneous coordinates in camera space
    homogeneous_cam_coordinates_3D = torch.cat([cam_0_coordinates, torch.ones(1, dtype=torch.float64, device="cuda:0")])
    # Step 5: Multiply by the camera-to-world matrix to get world coordinates
    world_coordinates_verify = torch.matmul(homogeneous_cam_coordinates_3D, cam_0_to_world.T)
    print("Deprojected 3D world coordinates:", world_coordinates_verify)
    homogeneous_world_coordinates = torch.cat(
        [world_coordinates_verify, torch.tensor([1.0], device="cuda:0", dtype=torch.float64)]
    )
    camera_coordinates = transform_world_to_camera_test(homogeneous_world_coordinates, cam_1_to_world)
    z = camera_coordinates[2]
    projected_coordinates_verify = camera_coordinates[:3] / z
    projected_coordinates_verify = torch.matmul(projected_coordinates_verify, intrinsics_test.T)
    x, y = projected_coordinates_verify[0], projected_coordinates_verify[1]
    print("Projected coordinates verify:", x, y)


def transform_world_to_camera_test(homogeneous_world_coordinates, cam_to_world_3x4):
    # Ensure the input tensors are on the same device and dtype
    device = homogeneous_world_coordinates.device
    dtype = homogeneous_world_coordinates.dtype

    cam_to_world_3x4 = cam_to_world_3x4.to(device).to(dtype)

    # Create a 4x4 version of the 3x4 camera-to-world matrix
    cam_to_world_4x4 = torch.zeros((4, 4), dtype=dtype, device=device)
    cam_to_world_4x4[:3, :] = cam_to_world_3x4
    cam_to_world_4x4[3, 3] = 1.0

    # Compute the inverse transformation matrix
    rotational_part = cam_to_world_4x4[:3, :3]
    translational_part = cam_to_world_4x4[:3, 3]
    inverse_rotation = rotational_part.T
    inverse_translation = torch.matmul(-inverse_rotation, translational_part)
    world_to_cam_4x4 = torch.zeros((4, 4), dtype=dtype, device=device)
    world_to_cam_4x4[:3, :3] = inverse_rotation
    world_to_cam_4x4[:3, 3] = inverse_translation
    world_to_cam_4x4[3, 3] = 1.0

    # Transform the point from world to camera coordinates
    homogeneous_camera_coordinates = torch.matmul(world_to_cam_4x4, homogeneous_world_coordinates)

    # Normalize to get the x, y, z coordinates in camera coordinates
    camera_coordinates = homogeneous_camera_coordinates[:3] / homogeneous_camera_coordinates[3]

    return camera_coordinates

cam_0_to_world = torch.tensor(
    [
        [-0.3407, 0.0572, -0.9384, 1.1704],
        [0.9397, 0.0525, -0.3380, 1.39677],
        [0.0299, -0.9970, -0.0717, -2.5307],
    ],
    device="cuda:0",
).double()  # This is used for testing

cam_1_to_world = torch.tensor(
    [
        [-0.4591, 0.1727, -0.8714, 1.2658],
        [0.8881, 0.1157, -0.4449, 1.204033],
        [0.0240, -0.9782, -0.2065, -2.4878],
    ],
    device="cuda:0",
).double()  # This is used for testing