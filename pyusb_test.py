#%%
import usb.core
import usb.util as util

import libusb_package
import usb.core
import usb.backend.libusb1
import sys
import time

libusb1_backend = usb.backend.libusb1.get_backend(find_library=libusb_package.find_library)

# find our device
dev = usb.core.find(idVendor=0x28de, idProduct=0x2101)

# was it found?
if dev is None:
    raise ValueError('Device not found')

bmRequestType = util.build_request_type(
                        util.CTRL_OUT,
                        util.CTRL_TYPE_CLASS,
                        util.CTRL_RECIPIENT_INTERFACE)



payload = [
    0xff,
    0x96,
    0x10,
    0xbe,
    0x5b,
    0x32,
    0x54,
    0x11,
    0xcf,
    0x83,
    0x75,
    0x53,
    0x8a,
    0x08,
    0x6a,
    0x53,
    0x58,
    0xd0,
    0xb1
]

for i in range(64-len(payload)):
    payload.append(0x00)

assert len(payload) == 64


wValue = 0x300 | payload[0]

# dev.ctrl_transfer(bmRequestType, 0x01, wValue, 0, 256)


good = False
while not good:
    try:
        dev.ctrl_transfer(bmRequestType, 0x09, wValue, 0, payload)
        good = True
    except:
        pass
    time.sleep(0.1)

for cfg in dev:
    sys.stdout.write(str(cfg.bConfigurationValue) + '\n')
    for intf in cfg:
        sys.stdout.write('\t' + \
                         str(intf.bInterfaceNumber) + \
                         ',' + \
                         str(intf.bAlternateSetting) + \
                         '\n')
        for ep in intf:
            sys.stdout.write('\t\t' + \
                             str(ep.bEndpointAddress) + \
                             '\n')

# set the active configuration. With no arguments, the first
# configuration will be the active one
dev.set_configuration()

# get an endpoint instance
cfg = dev.get_active_configuration()
intf = cfg[(0,0)]

ep = usb.util.find_descriptor(
    intf,
    # match the first OUT endpoint
    find_all=True,
    custom_match = \
    lambda e: \
        usb.util.endpoint_direction(e.bEndpointAddress) == \
        usb.util.ENDPOINT_OUT)

# assert ep is not None
# written = False
# while not written:
#     for e in ep:
#         try:
#             e.write(bytes.fromhex('0400000000'))
#             e.
#             written = True
#         except:
#             pass
for e in ep:
    print(e)
    # time.sleep(20)

ep = usb.util.find_descriptor(
    intf,
    # match the first OUT endpoint
    find_all=True,
    custom_match = \
    lambda e: \
        usb.util.endpoint_direction(e.bEndpointAddress) == \
        usb.util.ENDPOINT_IN)


for e in ep:
    print(e)

# # assert ep is not None

# reads = 1000
# ep = next(ep)
# while reads > 0:
#     a = ep.read(52, 100) 
#     print(a)
#     print(reads)
#     reads -= 1


# # # write the data
# # ep.write(bytes.fromhex(hex_string))