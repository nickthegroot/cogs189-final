import asyncio
import serial_asyncio as serial
import reactivex as rx
import reactivex.operators as ops

# Byte codes
CONNECT              = b'\xc0'
DISCONNECT           = b'\xc1'
AUTOCONNECT          = b'\xc2'
SYNC                 = b'\xaa'
EXCODE               = b'\x55'
POOR_SIGNAL          = b'\x02'
ATTENTION            = b'\x04'
MEDITATION           = b'\x05'
BLINK                = b'\x16'
HEADSET_CONNECTED    = b'\xd0'
HEADSET_NOT_FOUND    = b'\xd1'
HEADSET_DISCONNECTED = b'\xd2'
REQUEST_DENIED       = b'\xd3'
STANDBY_SCAN         = b'\xd4'
RAW_VALUE            = b'\x80'
ASIC_EEG_POWER       = b'\x83'

# Status codes
STATUS_CONNECTED = "connected"
STATUS_SCANNING = "scanning"
STATUS_STANDBY = "standby"

Packet = list[int]

class HeadsetProtocol(asyncio.Protocol):
    def __init__(self, val_checksum = True) -> None:
        super().__init__()
        self.val_checksum = val_checksum
        self.packets: rx.Subject[Packet] = rx.Subject()

    def connection_made(self, transport: serial.SerialTransport) -> None:
        transport.write(AUTOCONNECT)
        print('Connected! Waiting on signal...')
        return super().connection_made(transport)

    def data_received(self, data):
        while data:
            sync1, sync2, *data = data
            # Wait for sync data
            if sync1 == sync2 == ord(SYNC):
                plength, *data = data
                *packet, expected_checksum = data[:plength + 1]
                data = data[plength + 1:]

                if self.val_checksum:
                    assert expected_checksum == self.__calc_checksum(packet)
                    
                self.packets.on_next(packet)

    def __calc_checksum(self, packet: Packet):
        calc_checksum = sum(packet)
        calc_checksum = ~calc_checksum & 0xff
        return calc_checksum

    def connection_lost(self, exc):
        self.packets.on_completed()


class Headset:
    def __init__(self) -> None:
        self.poor_signal: rx.Subject[int] = rx.Subject()
        self.attention: rx.Subject[int] = rx.Subject()
        self.meditation: rx.Subject[int] = rx.Subject()
        self.blink: rx.Subject[int] = rx.Subject()
        self.raw_value: rx.Subject[int] = rx.Subject()
        self.waves: rx.Subject[dict[str, int]] = rx.Subject()

        self.__packet_map: dict[bytes, rx.Subject] = {
            POOR_SIGNAL: self.poor_signal,
            ATTENTION: self.attention,
            MEDITATION: self.meditation,
            BLINK: self.blink,
            RAW_VALUE: self.raw_value,
            ASIC_EEG_POWER: self.waves,
        }

    async def connect(self, port: str):
        self.protocol = HeadsetProtocol()
        self.protocol.packets.subscribe(self.handle_packet)

        self.transport, _ = await serial.create_serial_connection(
            asyncio.get_event_loop(),
            lambda: self.protocol,
            port,
            baudrate=115200,
            timeout=10,
        )

        await self.__await_connection()

    async def __await_connection(self):
        return await self.poor_signal.pipe(
            # Wait for headset to steady down
            ops.filter(lambda x: x < 5),
            ops.take(1),
        )

    def handle_packet(self, packet: Packet):
        while packet:
            code, *packet = packet
            code = bytes([code])
            if code == EXCODE:
                continue
            # Single Byte Codes
            if code < b"0x80":
                if code in self.__packet_map:
                    value, *packet = packet
                    self.__packet_map[code].on_next(value)
            # Multi-Byte code
            else:
                vlength, *packet = packet
                value, packet = packet[:vlength], packet[vlength:]
                # FIX: accessing value crashes elseway
                if code == RAW_VALUE:
                    raw = value[0] * 256 + value[1]
                    if raw >= 32768:
                        raw = raw - 65536
                    self.raw_value.on_next(raw)
                elif code == ASIC_EEG_POWER:
                    waves = {}
                    for i, wave in enumerate([
                        "delta",
                        "theta",
                        "low-alpha",
                        "high-alpha",
                        "low-beta",
                        "high-beta",
                        "low-gamma",
                        "mid-gamma",
                    ]):
                        j = i * 3
                        waves[wave] = (
                            value[j] * 255 * 255 + value[j + 1] * 255 + value[j + 2]
                        )
                    self.waves.on_next(waves)
            

    def disconnect(self):
        assert self.transport
        self.transport.write(DISCONNECT)
        self.transport.flush()
        self.transport.close()

        del self.transport
        del self.protocol
