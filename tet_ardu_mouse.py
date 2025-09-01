import time
import ctypes
from ctypes import wintypes
from arduino_test import ArduinoAbsMouse

# Constants
WH_MOUSE_LL = 14
WM_LBUTTONDOWN = 0x0201

# Load libraries
user32 = ctypes.WinDLL('user32', use_last_error=True)
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

# Define HHOOK as a HANDLE
HHOOK = wintypes.HANDLE

# Set argument types and return types for the functions we use
user32.SetWindowsHookExW.argtypes = [wintypes.INT, wintypes.LPVOID, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype = HHOOK

user32.CallNextHookEx.argtypes = [HHOOK, wintypes.INT, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype = wintypes.LPARAM

user32.UnhookWindowsHookEx.argtypes = [HHOOK]
user32.UnhookWindowsHookEx.restype = wintypes.BOOL

user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
user32.GetMessageW.restype = wintypes.INT

user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.TranslateMessage.restype = wintypes.BOOL

user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.DispatchMessageW.restype = wintypes.LPARAM

# Define the low-level mouse proc callback type
LowLevelMouseProc = ctypes.WINFUNCTYPE(
    wintypes.LPARAM,  # LRESULT (return)
    wintypes.INT,     # int nCode
    wintypes.WPARAM,  # WPARAM wParam
    wintypes.LPARAM   # LPARAM lParam
)

start_time = None
hook_handle = None

def low_level_mouse_handler(nCode, wParam, lParam):
    global start_time
    if nCode == 0 and wParam == WM_LBUTTONDOWN:
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        print(f"Latency: {latency_ms:.3f} ms")
        # Post a quit message to exit the message loop
        user32.PostQuitMessage(0)
    # Call next hook in the chain
    return user32.CallNextHookEx(hook_handle, nCode, wParam, lParam)

def main():
    global start_time, hook_handle

    # Create the callback function
    hook_proc = LowLevelMouseProc(low_level_mouse_handler)

    # Set the global low-level mouse hook with NULL module handle and thread ID 0
    hook_handle = user32.SetWindowsHookExW(WH_MOUSE_LL, hook_proc, None, 0)
    if not hook_handle:
        error_code = ctypes.get_last_error()
        raise OSError(f"SetWindowsHookEx failed with error code: {error_code}")

    # Initialize Arduino mouse simulation
    mouse = ArduinoAbsMouse()

    # Wait briefly to ensure hook is installed
    time.sleep(0.5)

    # Trigger simulated click and record start time
    start_time = time.perf_counter()
    # mouse.move_mouse(0.0, 0.0, 1)

    print("Moved Mouse")

    # Run a Windows message loop to intercept events
    msg = wintypes.MSG()
    while True:
        mouse.move_mouse(0.5, 0.5, 1)
        print("Moved Mouse2")
        mouse.move_mouse(0.5, 0.5, 2)
        ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
        if ret == 0:   # WM_QUIT
            break
        elif ret == -1:
            print("GetMessageW error")
            break
        else:
            mouse.move_mouse(0.5, 0.5, 1)
            print("4")
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
            print("5")

        time.sleep(0.001)
        print("Loop")

    # Unhook before exit
    if hook_handle:
        user32.UnhookWindowsHookEx(hook_handle)

if __name__ == "__main__":
    # Ensure you run this script as an Administrator
    main()
