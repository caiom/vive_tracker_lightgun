#%%
from arduino_test import ArduinoMouse2
from pynput.mouse import Controller
import time

mouse_pos = Controller()
mouse = ArduinoMouse2(1)

mouse.update_position(mouse_pos.position[0], mouse_pos.position[1])
print(mouse_pos.position)
mouse.move_mouse_absolute(2000, 1300, max_loops=30)
time.sleep(1)
print(mouse_pos.position)
mouse.close()
#%%


from arduino_test import ArduinoMouse2
from receive_udp import MAMECursor
import time



mouse_pos = MAMECursor(2560, 1440)
mouse = ArduinoMouse2(0.9486)
time.sleep(5)
original_pos = mouse_pos.get_position()
pos = mouse_pos.get_position()
mouse.update_position(pos[0], pos[1])
print(pos)

print(pos[0]-300, pos[1]-300)
# for i in range(100):
#     mouse.update_position(pos[0], pos[1])
#     mouse.move_mouse_absolute(pos[0]+20, pos[1]+20, max_loops=30)
#     time.sleep(0.1)

mouse.move_mouse_absolute(pos[0]-300, pos[1]-300, max_loops=30)
time.sleep(1)

pos = mouse_pos.get_position()
print(pos)
mouse.close()
mouse_pos.close()

print(f'Delta X: {pos[0] - original_pos[0]}, Y: {pos[1] - original_pos[1]}')




#%%
from arduino_test import ArduinoMouse2
from pynput.mouse import Controller
import time

mouse_pos = Controller()
mouse = ArduinoMouse2(1)
time.sleep(10)

for i in range(10):
    mouse.update_position(mouse_pos.position[0], mouse_pos.position[1])
    print(mouse_pos.position)
    mouse.move_mouse_absolute(mouse_pos.position[0], mouse_pos.position[1]+ 25, max_loops=30)
    time.sleep(0.2)
time.sleep(1)
print(mouse_pos.position)
mouse.close()