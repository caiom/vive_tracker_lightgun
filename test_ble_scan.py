import asyncio
from bleak import BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic
import struct

address = "18:8B:0E:2C:A6:AA"
MODEL_NBR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

def notification_handler(characteristic: BleakGATTCharacteristic, data: bytearray):
    """Simple notification handler which prints the data received."""
    button_state, = struct.unpack('<I', data)
    
    # Extract individual button states using bitwise operations
    trigger = not (button_state & 0x1)               # Bit 0
    button_a = not ((button_state >> 2) & 0x1)       # Bit 2
    button_b = not ((button_state >> 3) & 0x1)       # Bit 3

    print(f"Trigger: {'Pressed' if trigger else 'Released'}, "
          f"Button A: {'Pressed' if button_a else 'Released'}, "
          f"Button B: {'Pressed' if button_b else 'Released'}")

async def main(address):
    async with BleakClient(address) as client:
        await client.start_notify(MODEL_NBR_UUID, notification_handler)
        print("Listening for notifications... Press Ctrl+C to exit.")
        try:
            while True:
                await asyncio.sleep(1)  # Keep the program running
        except KeyboardInterrupt:
            print("Stopping notifications...")
        
        # Stop receiving notifications
        await client.stop_notify(MODEL_NBR_UUID)
        print("Disconnected.")

asyncio.run(main(address))