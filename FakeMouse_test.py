#%%
import FakeMouse
import time
import numpy as np

SCREEN_WIDTH = 2560 - 1
SCREEN_HEIGHT = 1440 - 1

mouse = FakeMouse.Mouse()
mouse.initialize()

while True:
    num = int(input("Enter a number: ")) # Taking input from user
    st = time.time()
    mouse.moveCursor(num, 1000)
    print(f'{time.time() - st}')

