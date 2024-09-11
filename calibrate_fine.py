import pygame
import triad_openvr
import numpy as np

# Triad Init
v = triad_openvr.triad_openvr()
v.print_discovered_objects()

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
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
               5: (1/4*screen.get_width(), 3/4*screen.get_height()),
               6: (1/4*screen.get_width(), 1/4*screen.get_height()),
               7: (3/4*screen.get_width(), 1/4*screen.get_height()),
               8: (3/4*screen.get_width(), 3/4*screen.get_height()),}

ball_pos = pygame.Vector2(0, screen.get_height())
pos_mapping = {0: (0, screen.get_height()-0),
               1: (0, 0),
               2: (screen.get_width()-0, 0),
               3: (screen.get_width()-0, screen.get_height()-0),
               4: (screen.get_width() / 2, screen.get_height() / 2),
               5: (1/4*screen.get_width(), 3/4*screen.get_height()),
               6: (1/4*screen.get_width(), 1/4*screen.get_height()),
               7: (3/4*screen.get_width(), 1/4*screen.get_height()),
               8: (3/4*screen.get_width(), 3/4*screen.get_height()),}


ball_pos = pygame.Vector2(large_ball_size, screen.get_height()-large_ball_size)
pos_mapping = {0: (large_ball_size, screen.get_height()-large_ball_size),
               1: (large_ball_size, large_ball_size),
               2: (screen.get_width()-large_ball_size, large_ball_size),
               3: (screen.get_width()-large_ball_size, screen.get_height()-large_ball_size),
               4: (screen.get_width() / 2, screen.get_height() / 2)}

ball_pos = pygame.Vector2(0, screen.get_height())
pos_mapping = {0: (0, screen.get_height()),
               1: (0, 0),
               2: (screen.get_width(), 0),
               3: (screen.get_width(), screen.get_height()),
               4: (screen.get_width() / 2, screen.get_height() / 2)}

screen_corners = []
corner_number = 0
num_key_points = len(pos_mapping)

trigger_pressed = False

while running:
    
    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    pose_matrix = eval(str(v.devices["tracker_1"].get_pose_matrix()))
    pose_matrix = np.asarray(pose_matrix)
    # print(v.devices["tracker_1"].get_pose_euler())
    controller_input = v.devices["tracker_1"].get_controller_inputs()
    # print(controller_input)
    for actionData in controller_input:
        print(f"{actionData.bActive}, {actionData.bChanged}, {actionData.bState}")

    # if controller_input['menu_button'] and not trigger_pressed:
    #     np.save(f'pose_matrix_{corner_number}.npy', pose_matrix)
    #     corner_number+=1
    #     ball_pos.x = pos_mapping[corner_number % num_key_points][0]
    #     ball_pos.y = pos_mapping[corner_number % num_key_points][1]
    #     trigger_pressed = True

    # if not controller_input['menu_button'] and trigger_pressed:
    #     trigger_pressed = False
    

    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
            running=False

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

pygame.quit()