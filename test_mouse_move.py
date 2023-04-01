#%%
from pynput.mouse import Button, Controller
import time
mouse = Controller()

while True:
    time.sleep(1)
    print('The current pointer position is {0}'.format(
        mouse.position))