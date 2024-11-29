import asyncio
import threading
from bleak import BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic
import struct
from typing import Optional, Tuple
import time


class LightGunInput:
    MODEL_NBR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

    def __init__(self, address="18:8B:0E:2C:A6:AA"):
        """
        Initializes the LightGunInput class and starts the background thread.
        
        Args:
            address (str): The MAC address of the BLE device.
        """
        self.address = address
        self._button_state: Optional[Tuple[bool, bool, bool]] = (False, False, False)
        self._connected = False
        self._lock = threading.Lock()

        # Event to signal the background thread to stop
        self._stop_event = threading.Event()

        # Start the background thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def notification_handler(
        self, characteristic: BleakGATTCharacteristic, data: bytearray
    ):
        """Handles incoming notifications and updates the button state."""
        try:
            button_state, = struct.unpack('<I', data)
            
            # Extract individual button states using bitwise operations
            trigger = not (button_state & 0x1)               # Bit 0
            button_a = not ((button_state >> 2) & 0x1)       # Bit 2
            button_b = not ((button_state >> 3) & 0x1)       # Bit 3

            with self._lock:
                self._button_state = (trigger, button_a, button_b)
        except struct.error as e:
            print(f"Failed to unpack data: {e}")

    def _run_loop(self):
        """Runs the asyncio event loop in a separate thread."""
        asyncio.run(self._async_main())

    async def _async_main(self):
        """Main coroutine that manages connection and reconnection logic."""
        while not self._stop_event.is_set():
            try:
                async with BleakClient(self.address) as client:
                    self._connected = client.is_connected()
                    if self._connected:
                        print(f"Connected to {self.address}")
                        await client.start_notify(
                            self.MODEL_NBR_UUID, self.notification_handler
                        )
                        print("Subscribed to notifications.")
                        
                        # Keep the connection open until disconnected or stop event is set
                        while not self._stop_event.is_set():
                            if not client.is_connected:
                                print("Connection lost.")
                                break
                            await asyncio.sleep(1)  # Check connection status periodically

                        # Stop notifications before disconnecting
                        await client.stop_notify(self.MODEL_NBR_UUID)
                        print("Unsubscribed from notifications.")
            except Exception as e:
                print(f"Connection or subscription failed: {e}")

            if not self._stop_event.is_set():
                print("Retrying connection in 5 seconds...")
                await asyncio.sleep(5)  # Wait before retrying

        print("Async main loop terminated.")

    def get_input(self) -> Optional[Tuple[bool, bool, bool]]:
        """
        Returns the current state of the buttons.
        
        Returns:
            A tuple (trigger, button_a, button_b) if connected and data is available, else None.
        """
        with self._lock:
            return self._button_state if self._connected else None

    def cleanup(self):
        """
        Stops the background thread and cleans up resources.
        """
        print("Cleaning up LightGunInput...")
        self._stop_event.set()
        self._thread.join()
        print("LightGunInput cleaned up.")


# Example Usage
if __name__ == "__main__":
    import sys

    def main():
        light_gun = LightGunInput()

        try:
            while True:
                input_state = light_gun.get_input()
                if input_state:
                    trigger, button_a, button_b = input_state
                    print(f"Trigger: {'Pressed' if trigger else 'Released'}, "
                          f"Button A: {'Pressed' if button_a else 'Released'}, "
                          f"Button B: {'Pressed' if button_b else 'Released'}")
                else:
                    print("Not connected or no input received yet.")
                time.sleep(1)  # Polling interval
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            light_gun.cleanup()
            sys.exit(0)

    main()