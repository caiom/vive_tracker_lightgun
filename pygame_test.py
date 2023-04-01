import pygame
import triad_openvr
import time
import sys
import math
import numpy as np
from sympy import Line, Plane, Segment, Point3D
from skspatial.objects import Point as ssPoint
from skspatial.objects import Line as ssLine
import time
from arduino_test import ArduinoMouse


def close_points(l1, l2):
    """return s and t, the parametric location of the closest
    points on l1 and l2, the actual points, and the separation between
    those two points.

    NOTE: these were checked on wolframalpha.com with query, e.g.
    "closest points on Line((0,0,0),(1,1,1)) and Line((0,1,0),(1,2,3))"

    Examples
    ========

    >>> from sympy import Line
    >>> close_points(Line((0, 0, 0), (1, 1, 1)), Line((0, 1, 0), (1, 2, 3)))
    (3/4, 1/4, Point3D(3/4, 3/4, 3/4), Point3D(1/4, 5/4, 3/4), sqrt(2)/2)
    >>> close_points(Line((0, 0, 0), (1, 2, 3)), Line((1, 1, 1), (2, 3, 5)))
    (7/5, 4/5, Point3D(7/5, 14/5, 21/5), Point3D(9/5, 13/5, 21/5), sqrt(5)/5)
    """
    # L1 = lambda s: l1.p1+s*l1.direction
    # L2 = lambda t: l2.p1+t*l2.direction
    # eq = lambda l1: Matrix(L1(var('s')) - L2(var('t'))).dot(l1.direction)
    # stdict = solve((
    #     eq(Line(var('x1 y1 z1'),var('x2 y2 z2'))),
    #     eq(Line(var('x3 y3 z3'),var('x4 y4 z4')))),
    #     var('s t'), dict=True)
    x1,y1,z1 = l1.p1
    x2,y2,z2 = l1.p2
    x3,y3,z3 = l2.p1
    x4,y4,z4 = l2.p2
    x21, y21, z21 = l1.direction
    x43, y43, z43 = l2.direction
    x31, y31, z31 = l2.p1 - l1.p1
    R1 = x21**2 + y21**2 + z21**2
    R2 = x43**2 + y43**2 + z43**2
    d4321 = x21*x43 + y21*y43 + z21*z43
    d3121 = x21*x31 + y21*y31 + z21*z31
    d4331 = x31*x43 + y31*y43 + z31*z43
    den = d4321**2 - R1*R2
    # R1*s - d4321*t - d3121 = 0
    # d4321*s - R2*t - d4331 = 0
    s = (d4321*d4331 - R2*d3121)/den
    t = (R1*d4331 - d4321*d3121)/den
    ps = l1.p1 + s*l1.direction
    pt = l2.p1 + t*l2.direction
    return s, t, ps.evalf(), pt.evalf(), ps.distance(pt)

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

v = triad_openvr.triad_openvr()
v.print_discovered_objects()

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0
mouse = ArduinoMouse(2560, 1440, 2, 1.74)

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
screen_corners = []
calibrated = False

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
            print((tracker_position, tracker_direction))
            screen_corners.append((tracker_position, tracker_direction))

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("purple")


    pose_matrix = eval(str(v.devices["tracker_1"].get_pose_matrix()))
    pose_matrix = np.asarray(pose_matrix)
    tracker_position = np.copy(pose_matrix[:, -1])
    pose_matrix[:, -1] = 0.0
    tracker_direction = pose_matrix @ np.asarray([[0], [0], [-1.0], [1.0]])
    tracker_direction = tracker_direction[:, 0]
    tracker_direction /= np.linalg.norm(tracker_direction)

    # keys = pygame.key.get_pressed()

    # if keys[pygame.K_w]:
    #     player_pos.y -= 300 * dt
    # if keys[pygame.K_s]:
    #     player_pos.y += 300 * dt
    # if keys[pygame.K_a]:
    #     player_pos.x -= 300 * dt
    # if keys[pygame.K_d]:
    #     player_pos.x += 300 * dt

    # if  keys[pygame.K_c]:
    #     screen_corners.append((tracker_position, tracker_direction))

    if len(screen_corners) == 8 and not calibrated:


        print('down-left')
        l1 = Line(tuple(screen_corners[0][0]), tuple(screen_corners[0][0] + screen_corners[0][1] * 5))
        l2 = Line(tuple(screen_corners[4][0]), tuple(screen_corners[4][0] + screen_corners[4][1] * 5))
        print(l1, l2)

        print(close_points(l1, l2))
        _, _, down_left_p, _, _ = close_points(l1, l2)

        print('top-left')
        l1 = Line(tuple(screen_corners[1][0]), tuple(screen_corners[1][0] + screen_corners[1][1] * 5))
        l2 = Line(tuple(screen_corners[5][0]), tuple(screen_corners[5][0]+ screen_corners[5][1] * 5))
        print(l1, l2)
        print(close_points(l1, l2))
        _, _, top_left_p, _, _ = close_points(l1, l2)

        print('top-right')
        l1 = Line(tuple(screen_corners[2][0]), tuple(screen_corners[2][0] + screen_corners[2][1] * 5))
        l2 = Line(tuple(screen_corners[6][0]), tuple(screen_corners[6][0]+ screen_corners[6][1] * 5))
        print(l1, l2)
        print(close_points(l1, l2))
        _, _, top_right_p, _, _ = close_points(l1, l2)

        print('down-right')
        l1 = Line(tuple(screen_corners[3][0]), tuple(screen_corners[3][0] + screen_corners[3][1] * 5))
        l2 = Line(tuple(screen_corners[7][0]), tuple(screen_corners[7][0] + screen_corners[7][1] * 5))
        print(l1, l2)
        print(close_points(l1, l2))

        calibrated = True
        screen_plane = Plane(down_left_p, top_left_p, top_right_p)
        down_left_p_np = np.array(down_left_p).astype(float)
        top_left_p_np = np.array(top_left_p).astype(float)
        top_right_p_np = np.array(top_right_p).astype(float)
        u_seg = ssLine(point=top_left_p_np, direction=top_right_p_np-top_left_p_np)
        v_seg = ssLine(point=top_left_p_np, direction=down_left_p_np-top_left_p_np) 
        print(screen_plane)
        # print(u_seg.evalf())
        # print(v_seg.evalf())


        


    if calibrated:
        st = time.time()
        # ray = Line(tuple(tracker_position), tuple(tracker_position + tracker_direction * 5))
        print(' t0' , time.time()-st)
        # p_screen = screen_plane.intersection(ray)[0]
        # print(p_screen.evalf())

        epsilon=1e-6

        #Define plane
        planeNormal = np.array(screen_plane.normal_vector).astype(float)
        planePoint = np.array(screen_plane.p1).astype(float)

        #Define ray
        rayDirection = tracker_direction
        rayPoint = tracker_position #Any point along the ray

        ndotu = planeNormal.dot(rayDirection) 

        if abs(ndotu) < epsilon:
            print ("no intersection or line is within plane")

        w = rayPoint - planePoint
        si = -planeNormal.dot(w) / ndotu
        Psi = w + si * rayDirection + planePoint
        Psi = rayPoint + (si*rayDirection)

        print ("intersection at", Psi)
        print(' t11' , time.time()-st)
        p_screen = ssPoint([Psi[0], Psi[1], Psi[2]])


        u_coord = u_seg.project_point(p_screen).distance_point(top_left_p_np) / u_seg.direction.norm()
        v_coord = v_seg.project_point(p_screen).distance_point(top_left_p_np) / v_seg.direction.norm()


        # print(' t1' , time.time()-st)
        # print(p_screen.evalf())
        # u_seg_proj = u_seg.projection(p_screen)
        # print(' t2' , time.time()-st)
        # u_seg_seg = Segment(top_left_p, u_seg_proj)
        # u_seg_length = u_seg_seg.length
        # u_coord = float(u_seg_length / u_seg.length)
        # v_coord = float((Segment(top_left_p, v_seg.projection(p_screen)).length / v_seg.length))
        # print(' t3' , time.time()-st)


        # print(p_screen.evalf())
        # print(u_coord)
        # print(v_coord)
        mouse.move_mouse_absolute_perc(u_coord, v_coord)
        ball_x = int(u_coord * screen.get_width())
        ball_y = int(v_coord * screen.get_height())


        player_pos.y = ball_y
        player_pos.x = ball_x
        print(' t4' , time.time()-st)


    pygame.draw.circle(screen, "red", player_pos, 40)


    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    # dt = clock.tick(100) / 1000
    # print(dt)

pygame.quit()