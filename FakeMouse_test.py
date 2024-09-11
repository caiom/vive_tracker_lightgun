#%%
import FakeMouse
import time
import numpy as np

SCREEN_WIDTH = 2560 - 1
SCREEN_HEIGHT = 1440 - 1

mouse = FakeMouse.Mouse()
mouse.initialize()
# mouse.sendMouseReport()

# while True:
#     num = int(input("Enter a number: ")) # Taking input from user
#     st = time.time()
#     mouse.moveCursor(num, 1000)
#     print(f'{time.time() - st}')


#%%
from arduino_test import FakeMouseAbs


fake_mouse = FakeMouseAbs(1.0)
fake_mouse.move_mouse_absolute(100, 100)


#%%
# fake_mouse.move_mouse_absolute(1000, 1000)
fake_mouse.update_position(0, 0)
