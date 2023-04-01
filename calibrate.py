import json
import pygame
import triad_openvr
import numpy as np
from itertools import combinations

def closest_points_on_rays(o1, d1, o2, d2):

    # Calculate the vector connecting the origins
    w0 = o1 - o2

    # Calculate dot products
    A = np.dot(d1, d1)
    B = np.dot(d1, d2)
    C = np.dot(d2, d2)
    D = np.dot(d1, w0)
    E = np.dot(d2, w0)

    # Check if the rays are parallel (or almost parallel)
    denominator = A * C - B * B
    if abs(denominator) < 1e-6:
        # Rays are almost parallel
        # In this case, we can just return the origin points as the closest points
        return o1, o2

    # Calculate the scalar parameters s and t
    s = (B * E - C * D) / denominator
    t = (A * E - B * D) / denominator

    print(s)
    print(t)

    # Calculate the closest points P1 and P2
    P1 = o1 + s * d1
    P2 = o2 + t * d2

    return P1, P2

# Triad Init
v = triad_openvr.triad_openvr()
v.print_discovered_objects()

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

ball_pos = pygame.Vector2(0, screen.get_height())
pos_mapping = {0: (0, screen.get_height()),
               1: (0, 0),
               2: (screen.get_width(), 0),
               3: (screen.get_width(), screen.get_height())}
screen_corners = []
corner_number = 0

while running:
    
    # fill the screen with a color to wipe away anything from last frame
    screen.fill("purple")

    pose_matrix = eval(str(v.devices["tracker_1"].get_pose_matrix()))
    pose_matrix = np.asarray(pose_matrix)

    # The last column is the tracker translation (x, y, z)
    tracker_position = np.copy(pose_matrix[:, -1])

    # Zero the translation
    pose_matrix[:, -1] = 0.0

    # Ajust the rotation in the expected axis
    tracker_direction = pose_matrix @ np.asarray([[0], [0], [-1.0], [1.0]])
    tracker_direction = tracker_direction[:, 0]
    tracker_direction /= np.linalg.norm(tracker_direction)

    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:

            print((tracker_position, tracker_direction))
            corner_number += 1
            ball_pos.x = pos_mapping[corner_number % 4][0]
            ball_pos.y = pos_mapping[corner_number % 4][1]
            screen_corners.append((tracker_position, tracker_direction))

        elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:

            assert len(screen_corners) % 4 == 0, 'Calibration failed, please point the gun at the four corners at least twice'

            calibration_dict = {}
            for i, corner in enumerate(['bottom_left', 'top_left', 'top_right', 'bottom_right']):
                print(corner)
                corner_points = []
                for j in range(i, len(screen_corners), 4):
                    corner_points.append(screen_corners[j])
                
                point_comb = list(combinations(corner_points, 2))

                comb_point = np.zeros(3)
                for ray1, ray2 in point_comb:
                    p1, p2 = closest_points_on_rays(ray1[0], ray1[1], ray2[0], ray2[1])
                    mean_point = (p1 + p2) / 2
                    comb_point += mean_point

                calibration_dict[corner] = comb_point / len(point_comb)

            # Add plane normal
            calibration_dict['plane_normal'] = np.cross(calibration_dict['bottom_left'] - calibration_dict['top_left'], calibration_dict['top_right'] - calibration_dict['top_left'])

            # Convert to lists
            for key, val in calibration_dict.items():
                calibration_dict[key] = list(val)

            # Save
            with open("calibration.json", 'w') as f:
                json.dump(calibration_dict, f, indent=2)

            print('Calibration done')
            running=False

    # Draw circle as a reference for aiming
    pygame.draw.circle(screen, "red", ball_pos, 40)

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(100) / 1000
    print(dt)

pygame.quit()