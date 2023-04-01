#%%
import socket
import threading
import time
import struct

import numpy as np

# Configure the IP address and port of the C++ application
cpp_server_ip = "127.0.0.1"
cpp_server_port = 3374

# Create a UDP socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind(('127.0.0.1', 4437))
udp_socket.settimeout(2)

is_connected = False
lock = threading.Lock()
pos_x = 0.5
pos_y = 0.5
width = 288
height = 233

def send_udp_message():
    global is_connected
    global udp_socket
    # Set the two bytes to match the check in the C++ application: buffer[0] == 12 && buffer[1] == 37
    message = bytearray([12, 37])

    # Send the message to the C++ application
    udp_socket.sendto(message, (cpp_server_ip, cpp_server_port))
    if not is_connected:
        try:
            data, addr = udp_socket.recvfrom(18) # Buffer size is set to 18 bytes
            if data:
                print(' connected')
                is_connected = True
            else:
                print('Recreating socket')
                udp_socket.close()
                udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                udp_socket.bind(('127.0.0.1', 4437))
                udp_socket.settimeout(2)
        except:
                print('Recreating socket')
                udp_socket.close()
                udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                udp_socket.bind(('127.0.0.1', 4437))
                udp_socket.settimeout(2)


def udp_sender_thread():
    while True:
        send_udp_message()
        time.sleep(2)  # Sleep for 2 seconds


def receiver_thread():
    global pos_x, pos_y, width, height, is_connected
# Keep the main program running to allow the sender_thread to continue executing
    try:
        while True:
            if is_connected:
                try:
                    data, addr = udp_socket.recvfrom(18) # Buffer size is set to 18 bytes
                except:
                    is_connected = False
            else:
                time.sleep(1)
                continue

            if data:
                # Unpack the data using struct
                char1, char2, width, height, pos_x, pos_y = struct.unpack('<cciiff', data)

                # Convert the bytes for chars to strings
                char1 = char1.decode('utf-8')
                char2 = char2.decode('utf-8')
            else:
                break
    except KeyboardInterrupt:
        print("Exiting...")

# # Close the socket
# udp_socket.close()


class MAMECursor:
    def __init__(self, screen_width, screen_height):

        self.screen_width = screen_width
        self.screen_height = screen_height
        # Create and start the thread
        self.sender_thread = threading.Thread(target=udp_sender_thread)
        self.sender_thread.daemon = True  # Set the thread as a daemon so it will exit when the main program exits
        self.sender_thread.start()
        self.recv_thread = threading.Thread(target=receiver_thread)
        self.recv_thread.daemon = True  # Set the thread as a daemon so it will exit when the main program exits
        self.recv_thread.start()
        self.screen_4_3_width = (screen_height / 3) * 4
        self.screen_4_3_border_size = (screen_width - self.screen_4_3_width) / 2
        self.border_left = self.screen_4_3_border_size
        self.border_right = self.screen_4_3_border_size + self.screen_4_3_width

    def get_position(self):
        y = 1-pos_y
        x = pos_x
        x = x*self.screen_4_3_width + self.border_left
        y = y*self.screen_height
        x = np.clip(x, self.border_left, self.border_right)
        y = np.clip(y, 0, self.screen_height)
        return (x, y)
    
    def process_target(self, x, y):
        target_x = x*self.screen_width
        target_y = y*self.screen_height
        target_x = np.clip(target_x, 0, self.screen_width)
        target_y = np.clip(target_y, 0, self.screen_height)

        return (target_x, target_y)
    
    def close(self):
        self.sender_thread.join()
        self.recv_thread.join()
        udp_socket.close()

    

# mame_cursor = MAMECursor(2560, 1440)

# while True:
#     time.sleep(2)
#     print(mame_cursor.get_position())

# mame_cursor.close()

