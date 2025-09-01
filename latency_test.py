import pygame
import sys
from light_gun_input import LightGunInput

inp = LightGunInput()
# Initialize Pygame
pygame.init()

# Set up full-screen mode
screen_info = pygame.display.Info()
screen_width, screen_height = screen_info.current_w, screen_info.current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.display.set_caption("Full-Screen Red Circle Drawer")

# Define colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Circle settings
circle_radius = 50  # Adjust as needed

# List to store circle positions
circles = []

# Set up the clock for managing the frame rate
clock = pygame.time.Clock()
FPS = 200

ball_pos = pygame.Vector2(screen_width//2, screen_height//2)

def main():
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Press ESC to exit
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    pos = event.pos
                    circles.append(pos)

        ir = inp.get_input()
        t, b1, b2 = False, False, False
        if ir is not None:
            t, b1, b2 = ir

        if t or b1 or b2:
            circles.append(ball_pos)


        # Fill the screen with black
        screen.fill(BLACK)

        # Draw all circles
        for pos in circles:
            pygame.draw.circle(screen, RED, pos, circle_radius)

        # Update the display
        pygame.display.flip()

        # Tick the clock to maintain FPS
        clock.tick(FPS)

    # Quit Pygame
    inp.cleanup()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
