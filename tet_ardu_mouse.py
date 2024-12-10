#%%

from arduino_test import ArduinoAbsMouse
mouse = ArduinoAbsMouse()
mouse.move_mouse(0.0, 1.0, 0)
mouse.close()