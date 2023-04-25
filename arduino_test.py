# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:09:03 2019

@author: v3n0w
"""


#%%

import serial
import time
import struct
from pynput.mouse import Controller
import numpy as np
import ctypes
import threading


ntdll = ctypes.WinDLL('NTDLL.DLL')

NSEC_PER_SEC = 1000000000

class ArduinoMouse:
    def __init__(self, sensi: float):
        self.pos_x = 0
        self.pos_y = 0
        self.sensi = sensi
        self.target_x = 0.0
        self.target_y = 0.0
        self.max_error = 1.0
        self.set_resolution(1e-5)
        self.arduino = serial.Serial(port='COM3', baudrate=115200)
        self.running = True
        self.thread_lock = threading.Lock()
        self.thread = threading.Thread(target=self.move_mouse_thread)
        self.mouse_buttom = 0

    def move_mouse_thread(self):
        
        last_mouse_buttom = 0

        while self.running:

            with self.thread_lock:
                pos_x = self.pos_x
                pos_y = self.pos_y
                target_x = self.target_x
                target_y = self.target_y
                mouse_buttom = self.mouse_buttom
                sensi = self.sensi


            delta_x_pixels = target_x - pos_x
            delta_x = delta_x_pixels / sensi
            delta_y_pixels = target_y - pos_y
            delta_y = delta_y_pixels / sensi
            
            moviment_x = int(np.clip(delta_x, -125, 125))
            moviment_y = int(np.clip(delta_y, -125, 125))

            if moviment_x != 0 or moviment_y != 0 or mouse_buttom != last_mouse_buttom:

                self.move_mouse(moviment_x, moviment_y, mouse_buttom)
                last_mouse_buttom = mouse_buttom
                with self.thread_lock:
                    self.pos_x += moviment_x * (1/sensi)
                    self.pos_y += moviment_y * (1/sensi)
            
            time.sleep(0.01)


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
        self.arduino.write(struct.pack('<BiiBB', 37, x, y, mouse_buttom, mouse_buttom))

    def move_mouse_absolute(self, x:float, y:float, mouse_buttom:int=0, max_loops=10):

        if not self.thread.is_alive():
            self.thread.start()

        with self.thread_lock:
            self.target_x = x
            self.target_y = y
            self.mouse_buttom = mouse_buttom

    def update_position(self, x, y):
        with self.thread_lock:
            self.pos_x = x
            self.pos_y = y

    def set_sensi(self, sensi):
        with self.thread_lock:
            self.sensi = sensi

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.arduino.close()

class WindowsCursor:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width - 1
        self.screen_height = screen_height - 1
        self.mouse_win = Controller()
        self.screen_4_3_width = (screen_height / 3) * 4
        self.screen_4_3_border_size = (screen_width - self.screen_4_3_width) / 2
        self.border_left = self.screen_4_3_border_size
        self.border_right = self.screen_4_3_border_size + self.screen_4_3_width

    def get_position(self):
        return tuple(self.mouse_win.position)
    
    def process_target(self, x, y, mode='16:9'):
        target_x = x*self.screen_width
        target_y = y*self.screen_height

        if mode == '16:9':
            target_x = np.clip(target_x, 0, self.screen_width)
            target_y = np.clip(target_y, 0, self.screen_height)
        else:
            target_x = np.clip(target_x, self.border_left, self.border_right)
            target_y = np.clip(target_y, 0, self.screen_height)

        return (target_x, target_y)