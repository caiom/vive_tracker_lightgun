import asyncio
from bleak import BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic

address = "18:8B:0E:2C:A6:AA"
MODEL_NBR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

def notification_handler(characteristic: BleakGATTCharacteristic, data: bytearray):
    """Simple notification handler which prints the data received."""
    print("%s: %r", characteristic.description, data)

async def main(address):
    async with BleakClient(address) as client:
        await client.start_notify(MODEL_NBR_UUID, notification_handler)
        await asyncio.sleep(60.0)
        await client.stop_notify(MODEL_NBR_UUID)

asyncio.run(main(address))