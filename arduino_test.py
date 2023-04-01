# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:09:03 2019

@author: v3n0w
"""

#%%
# import ctypes
# import statistics
# import time


# ntdll = ctypes.WinDLL('NTDLL.DLL')

# NSEC_PER_SEC = 1000000000




# set_resolution(1e-5)
# #%%
# import serial
# import time
# import struct
# from pynput.mouse import Controller
# import numpy as np
# import math

# SCREEN_WIDTH = 2560 - 1
# SCREEN_HEIGHT = 1440 - 1
# MAX_ERROR = 2
# SENSI = 0.74

# mouse = Controller()
# arduino = serial.Serial(port='COM3', baudrate=115200)

# target_x = 0
# def move_mouse(x: int, y: int):
#     arduino.write(struct.pack('<ii', x, y))
#     time.sleep(0.000001)

# def reset_x_axis():
#     delta_x_pixels = -mouse.position[0]

#     while np.abs(delta_x_pixels) > 10:
#         moviment_x = int(np.clip(delta_x_pixels, -25, 25))
#         move_mouse(moviment_x, 0)
#         delta_x_pixels = -mouse.position[0]


# def reset_y_axis():
#     delta_y_pixels = -mouse.position[1]

#     while np.abs(delta_y_pixels) > 10:
#         moviment_y = int(np.clip(delta_y_pixels, -25, 25))
#         move_mouse(0, moviment_y)
#         delta_y_pixels = -mouse.position[1]

# move_dict_y = {}
# for i in range(127):
#     reset_y_axis()
#     pos_before = mouse.position[1]
#     move_mouse(0, i)
#     pos_after = mouse.position[1]
#     move_dict_y[i] = pos_after - pos_before
#     print(f'target: {i}, actual: {pos_after - pos_before}')



# while True:
#     num = int(input("Enter a number: ")) # Taking input from user
#     target_x = num
#     target_y = 1000

#     pos = mouse.position
#     delta_x_pixels = target_x - pos[0]
#     delta_x = delta_x_pixels / SENSI
#     delta_y_pixels = target_y - pos[1]
#     delta_y = delta_y_pixels / SENSI

#     delta_error = np.sqrt(delta_x_pixels**2 + delta_y_pixels**2)
#     st = time.time()

#     while delta_error > MAX_ERROR:
    
#         moviment_x = int(np.clip(delta_x, -125, 125))
#         moviment_y = int(np.clip(delta_y, -125, 125))
#         # print(moviment_x, moviment_y)
#         sm = time.time()
#         move_mouse(moviment_x, moviment_y)
#         print(f'{time.time() - sm}')
        
#         # print(value) # printing the value
#         pos = mouse.position
#         delta_x_pixels = target_x - pos[0]
#         delta_x = delta_x_pixels / SENSI
#         delta_y_pixels = target_y - pos[1]
#         delta_y = delta_y_pixels / SENSI
#         delta_error = np.sqrt(delta_x_pixels**2 + delta_y_pixels**2)
#         # print(delta_x_pixels, delta_y_pixels)
#         # print(delta_x, delta_y)
#         # print(f'ERROR: {delta_error}')
#     print(f'{time.time() - st}')

#%%

import serial
import time
import struct
from pynput.mouse import Controller
import numpy as np
import ctypes


ntdll = ctypes.WinDLL('NTDLL.DLL')

NSEC_PER_SEC = 1000000000

class ArduinoMouse:
    def __init__(self, screen_width: int, screen_height: int, max_error: float, sensi: float):
        self.screen_width = screen_width - 1
        self.screen_height = screen_height - 1
        self.max_error = max_error
        self.sensi = sensi
        self.target_x = 0.0
        self.target_y = 0.0
        self.mouse_win = Controller()
        self.arduino = serial.Serial(port='COM3', baudrate=115200)
        self.set_resolution(1e-5)
        self.screen_4_3_width = (screen_height / 3) * 4
        self.screen_4_3_border_size = (screen_width - self.screen_4_3_width) / 2
        self.border_left = self.screen_4_3_border_size
        self.border_right = self.screen_4_3_border_size + self.screen_4_3_width

    def set_resolution_ns(self, resolution):
        """Set resolution of system timer.

        See `NtSetTimerResolution`

        http://undocumented.ntinternals.net/index.html?page=UserMode%2FUndocumented%20Functions%2FTime%2FNtSetTimerResolution.html
        http://www.windowstimestamp.com/description
        https://bugs.python.org/issue13845

        """
        # NtSetTimerResolution uses 100ns units
        resolution = ctypes.c_ulong(int(resolution // 100))
        current = ctypes.c_ulong()

        r = ntdll.NtSetTimerResolution(resolution, 1, ctypes.byref(current))

        # NtSetTimerResolution uses 100ns units
        return current.value * 100
    
    def set_resolution(self, resolution):
        return self.set_resolution_ns(resolution * NSEC_PER_SEC) / NSEC_PER_SEC

    def move_mouse(self, x: int, y: int, mouse_buttom:int=0):
        self.arduino.write(struct.pack('<iiB', x, y, mouse_buttom))

    def move_mouse_absolute(self, x:float, y:float, mouse_buttom:int=0, max_loops=4):

        self.target_x = np.clip(x, 0, self.screen_width)
        self.target_y = np.clip(y, 0, self.screen_height)

        pos = list(self.mouse_win.position)
        # pos[0] = np.clip(pos[0], self.border_left, self.border_right)
        delta_x_pixels = self.target_x - pos[0]
        delta_x = delta_x_pixels / self.sensi
        delta_y_pixels = self.target_y - pos[1]
        delta_y = delta_y_pixels / self.sensi

        delta_error = np.sqrt(delta_x_pixels**2 + delta_y_pixels**2)
        loop = 0
        st = time.time()
        while delta_error > self.max_error and loop < max_loops:
        
            moviment_x = int(np.clip(delta_x, -125, 125))
            moviment_y = int(np.clip(delta_y, -125, 125))

            self.move_mouse(int(moviment_x), int(moviment_y), mouse_buttom)
            time.sleep(0.0001)

            pos = list(self.mouse_win.position)
            # pos[0] = np.clip(pos[0], self.border_left, self.border_right)
            delta_x_pixels = self.target_x - pos[0]
            delta_x = delta_x_pixels / self.sensi
            delta_y_pixels = self.target_y - pos[1]
            delta_y = delta_y_pixels / self.sensi
            delta_error = np.sqrt(delta_x_pixels**2 + delta_y_pixels**2)
            loop += 1

    def move_mouse_absolute_perc(self, x:float, y:float, mouse_buttom:int=0, max_loops=4):
        self.move_mouse_absolute(x*self.screen_width, y*self.screen_height, mouse_buttom, max_loops)



class ArduinoMouse2:
    def __init__(self, sensi: float):
        self.pos_x = 0
        self.pos_y = 0
        self.sensi = sensi
        self.target_x = 0.0
        self.target_y = 0.0
        self.max_error = 1.0
        self.arduino = serial.Serial(port='COM3', baudrate=115200)

    def move_mouse(self, x: int, y: int, mouse_buttom:int=0):
        self.arduino.write(struct.pack('<iiB', x, y, mouse_buttom))

    def move_mouse_absolute(self, x:float, y:float, mouse_buttom:int=0, max_loops=10):

        self.target_x = x
        self.target_y = y

        delta_x_pixels = self.target_x - self.pos_x
        delta_x = delta_x_pixels / self.sensi
        delta_y_pixels = self.target_y - self.pos_y
        delta_y = delta_y_pixels / self.sensi

        delta_error = np.sqrt(delta_x_pixels**2 + delta_y_pixels**2)
        loop = 0
        while delta_error > self.max_error and loop < max_loops:
        
            moviment_x = int(np.clip(delta_x, -125, 125))
            moviment_y = int(np.clip(delta_y, -125, 125))

            self.move_mouse(int(moviment_x), int(moviment_y), mouse_buttom)
            self.pos_x += moviment_x
            self.pos_y += moviment_y

            delta_x_pixels = self.target_x - self.pos_x
            delta_x = delta_x_pixels / self.sensi
            delta_y_pixels = self.target_y - self.pos_y
            delta_y = delta_y_pixels / self.sensi
            delta_error = np.sqrt(delta_x_pixels**2 + delta_y_pixels**2)
            loop += 1

    def update_position(self, x, y):
        self.pos_x = x
        self.pos_y = y

class WindowsCursor:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width - 1
        self.screen_height = screen_height - 1
        self.mouse_win = Controller()

    def get_position(self):
        return tuple(self.mouse_win.position)
    
    def process_target(self, x, y):
        target_x = x*self.screen_width
        target_y = y*self.screen_height
        target_x = np.clip(target_x, 0, self.screen_width)
        target_y = np.clip(target_y, 0, self.screen_height)

        return (target_x, target_y)



            


#%%

# mouse = ArduinoMouse(2560, 1440, 2, 0.74)

#%%
# mouse.move_mouse_absolute(0, 1000)



