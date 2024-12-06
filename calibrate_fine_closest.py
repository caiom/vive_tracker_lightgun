import pygame
import numpy as np
from pose_estimator_icam import PoseEstimator
from light_gun_input import LightGunInput

def calculate_pos_and_dir_from_pose(pose_matrix):
    tracker_position = np.copy(pose_matrix[:3, -1])
    tracker_direction = -np.copy(pose_matrix[:3, 2])

    return (tracker_position, tracker_direction)

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

    # Calculate the closest points P1 and P2
    P1 = o1 + s * d1
    P2 = o2 + t * d2

    return P1, P2

# Pose estimator
pose_estimator = PoseEstimator()

light_gun_input = LightGunInput()

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720), pygame.FULLSCREEN)
# screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

large_ball_size = 40
small_ball_size = 5

ball_pos = pygame.Vector2(large_ball_size, screen.get_height()-large_ball_size)
pos_mapping = {0: (large_ball_size, screen.get_height()-large_ball_size),
               1: (large_ball_size, large_ball_size),
               2: (screen.get_width()-large_ball_size, large_ball_size),
               3: (screen.get_width()-large_ball_size, screen.get_height()-large_ball_size),
               4: (screen.get_width() / 2, screen.get_height() / 2),
               5: (large_ball_size, 1/2*screen.get_height()),
               6: (screen.get_width()-large_ball_size, 1/2*screen.get_height()),
               7: (1/2*screen.get_width(), screen.get_height()-large_ball_size),}

# ball_pos = pygame.Vector2(0, screen.get_height())
# pos_mapping = {0: (0, screen.get_height()-0),
#                1: (0, 0),
#                2: (screen.get_width()-0, 0),
#                3: (screen.get_width()-0, screen.get_height()-0),
#                4: (screen.get_width() / 2, screen.get_height() / 2),
#                5: (1/4*screen.get_width(), 3/4*screen.get_height()),
#                6: (1/4*screen.get_width(), 1/4*screen.get_height()),
#                7: (3/4*screen.get_width(), 1/4*screen.get_height()),
#                8: (3/4*screen.get_width(), 3/4*screen.get_height()),}


# ball_pos = pygame.Vector2(large_ball_size, screen.get_height()-large_ball_size)
# pos_mapping = {0: (large_ball_size, screen.get_height()-large_ball_size),
#                1: (large_ball_size, large_ball_size),
#                2: (screen.get_width()-large_ball_size, large_ball_size),
#                3: (screen.get_width()-large_ball_size, screen.get_height()-large_ball_size),
#                4: (screen.get_width() / 2, screen.get_height() / 2)}

# ball_pos = pygame.Vector2(0, screen.get_height())
# pos_mapping = {0: (0, screen.get_height()),
#                1: (0, 0),
#                2: (screen.get_width(), 0),
#                3: (screen.get_width(), screen.get_height()),
#                4: (screen.get_width() / 2, screen.get_height() / 2)}

screen_corners = []
corner_number = 0
num_key_points = len(pos_mapping)

trigger_pressed = False
button_pressed = False
last_pose = None

font = pygame.font.Font(pygame.font.get_default_font(), 36)

tracker_to_gun = np.load("tracker_to_gun.npy")

while running:
    
    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    pose_matrix = pose_estimator.get_pose()
    if light_gun_input.get_input() is not None:
        _, button_pressed, _ = light_gun_input.get_input()

    if pose_matrix is not None:
        print("Valid")

    pose_matrix = tracker_to_gun @ pose_matrix

    if button_pressed and not trigger_pressed:
        # np.save(f'pose_matrix_{corner_number}.npy', pose_matrix)
        # corner_number+=1
        # ball_pos.x = pos_mapping[corner_number % num_key_points][0]
        # ball_pos.y = pos_mapping[corner_number % num_key_points][1]
        last_pose = np.copy(pose_matrix)
        trigger_pressed = True

    if not button_pressed and trigger_pressed:
        trigger_pressed = False

    if last_pose is not None:
        lo, ld = calculate_pos_and_dir_from_pose(last_pose)
        co, cd = calculate_pos_and_dir_from_pose(pose_matrix)

        p1, p2 = closest_points_on_rays(lo, ld, co, cd)

        mean_point = (p1 + p2) / 2
        dist = np.linalg.norm(p1-p2)

        text_surface = font.render(f'Dist: {dist:.1f} \n {mean_point[0]:.1f} \n {mean_point[1]:.1f} \n {mean_point[2]:.1f}', True, (200, 0, 0))
        screen.blit(text_surface, dest=(0,0))

        
    

    # poll for events
    # pygame.QUIT event means the user clicked X to close your window

    key_pressed = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
            running=False

        if event.type == pygame.KEYDOWN and event.key == pygame.K_k:
            key_pressed = True

    if key_pressed and not trigger_pressed:
        corner_number+=1
        ball_pos.x = pos_mapping[corner_number % num_key_points][0]
        ball_pos.y = pos_mapping[corner_number % num_key_points][1]

        trigger_pressed = True

    if key_pressed and trigger_pressed:
        trigger_pressed = False


    # Draw circle as a reference for aiming
    pygame.draw.circle(screen, "red", ball_pos, large_ball_size)
    pygame.draw.circle(screen, "black", ball_pos, small_ball_size)

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(100) / 1000
    # print(dt)

light_gun_input.cleanup()
pose_estimator.cleanup()
pygame.quit()