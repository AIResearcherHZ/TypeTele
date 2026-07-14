import atexit
import logging
import time
from typing import Optional, Sequence, Union, Tuple

import numpy as np
from ui import console

PROTOCOL_VERSION = 2.0

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POS_VEL_CUR = 126

LEN_PRESENT_POSITION = 4
LEN_PRESENT_VELOCITY = 4
LEN_PRESENT_CURRENT = 2
LEN_PRESENT_POS_VEL_CUR = 10
LEN_GOAL_POSITION = 4

DEFAULT_POS_SCALE = 2.0 * np.pi / 4096
DEFAULT_VEL_SCALE = 0.229 * 2.0 * np.pi / 60.0
DEFAULT_CUR_SCALE = 1.34


def dynamixel_cleanup_handler():
    open_clients = list(DynamixelClient.OPEN_CLIENTS)
    for open_client in open_clients:
        if open_client.port_handler.is_using:
            logging.warning("Forcing client to close.")
        open_client.port_handler.is_using = False
        open_client.disconnect()


def signed_to_unsigned(value: int, size: int) -> int:
    if value < 0:
        bit_size = 8 * size
        max_value = (1 << bit_size) - 1
        value = max_value + value
    return value


def unsigned_to_signed(value: int, size: int) -> int:
    bit_size = 8 * size
    if (value & (1 << (bit_size - 1))) != 0:
        value = -((1 << bit_size) - value)
    return value


class DynamixelClient:
    OPEN_CLIENTS = set()

    def __init__(
        self,
        motor_ids: Sequence[int],
        port: str = "/dev/ttyUSB0",
        baudrate: int = 1000000,
        lazy_connect: bool = False,
        pos_scale: Optional[float] = None,
        vel_scale: Optional[float] = None,
        cur_scale: Optional[float] = None,
    ):
        import dynamixel_sdk

        self.dxl = dynamixel_sdk

        self.motor_ids = list(motor_ids)
        self.port_name = port
        self.baudrate = baudrate
        self.lazy_connect = lazy_connect

        self.port_handler = self.dxl.PortHandler(port)
        self.packet_handler = self.dxl.PacketHandler(PROTOCOL_VERSION)

        self._pos_vel_cur_reader = DynamixelPosVelCurReader(
            self,
            self.motor_ids,
            pos_scale=pos_scale if pos_scale is not None else DEFAULT_POS_SCALE,
            vel_scale=vel_scale if vel_scale is not None else DEFAULT_VEL_SCALE,
            cur_scale=cur_scale if cur_scale is not None else DEFAULT_CUR_SCALE,
        )
        self._pos_reader = DynamixelPosReader(
            self,
            self.motor_ids,
            pos_scale=pos_scale if pos_scale is not None else DEFAULT_POS_SCALE,
            vel_scale=vel_scale if vel_scale is not None else DEFAULT_VEL_SCALE,
            cur_scale=cur_scale if cur_scale is not None else DEFAULT_CUR_SCALE,
        )
        self._vel_reader = DynamixelVelReader(
            self,
            self.motor_ids,
            pos_scale=pos_scale if pos_scale is not None else DEFAULT_POS_SCALE,
            vel_scale=vel_scale if vel_scale is not None else DEFAULT_VEL_SCALE,
            cur_scale=cur_scale if cur_scale is not None else DEFAULT_CUR_SCALE,
        )
        self._cur_reader = DynamixelCurReader(
            self,
            self.motor_ids,
            pos_scale=pos_scale if pos_scale is not None else DEFAULT_POS_SCALE,
            vel_scale=vel_scale if vel_scale is not None else DEFAULT_VEL_SCALE,
            cur_scale=cur_scale if cur_scale is not None else DEFAULT_CUR_SCALE,
        )
        self._sync_writers = {}

        self.OPEN_CLIENTS.add(self)

    @property
    def is_connected(self) -> bool:
        return self.port_handler.is_open

    def connect(self):
        assert not self.is_connected, "Client is already connected."

        if self.port_handler.openPort():
            logging.info("Succeeded to open port: %s", self.port_name)
        else:
            raise OSError(
                (
                    "Failed to open port at {} (Check that the device is powered "
                    "on and connected to your computer)."
                ).format(self.port_name)
            )

        if self.port_handler.setBaudRate(self.baudrate):
            logging.info("Succeeded to set baudrate to %d", self.baudrate)
        else:
            raise OSError(
                (
                    "Failed to set the baudrate to {} (Ensure that the device was "
                    "configured for this baudrate)."
                ).format(self.baudrate)
            )

    def disconnect(self):
        if not self.is_connected:
            return
        if self.port_handler.is_using:
            logging.error("Port handler in use; cannot disconnect.")
            return
        self.set_torque_enabled(self.motor_ids, False, retries=0)
        self.port_handler.closePort()
        if self in self.OPEN_CLIENTS:
            self.OPEN_CLIENTS.remove(self)

    def set_torque_enabled(
        self,
        motor_ids: Sequence[int],
        enabled: bool,
        retries: int = -1,
        retry_interval: float = 0.25,
    ):
        remaining_ids = list(motor_ids)
        while remaining_ids:
            remaining_ids = self.write_byte(
                remaining_ids,
                int(enabled),
                ADDR_TORQUE_ENABLE,
            )
            if remaining_ids:
                logging.error(
                    "Could not set torque %s for IDs: %s",
                    "enabled" if enabled else "disabled",
                    str(remaining_ids),
                )
            if retries == 0:
                break
            time.sleep(retry_interval)
            retries -= 1

    def read_pos_vel_cur(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._pos_vel_cur_reader.read()

    def read_pos(self) -> np.ndarray:
        return self._pos_reader.read()

    def read_vel(self) -> np.ndarray:
        return self._vel_reader.read()

    def read_cur(self) -> np.ndarray:
        return self._cur_reader.read()

    def write_desired_pos(self, motor_ids: Sequence[int], positions: np.ndarray):
        assert len(motor_ids) == len(positions)

        positions = positions / self._pos_vel_cur_reader.pos_scale
        self.sync_write(motor_ids, positions, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

    def write_byte(
        self,
        motor_ids: Sequence[int],
        value: int,
        address: int,
    ) -> Sequence[int]:
        self.check_connected()
        errored_ids = []
        for motor_id in motor_ids:
            comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, address, value
            )
            success = self.handle_packet_result(
                comm_result, dxl_error, motor_id, context="write_byte"
            )
            if not success:
                errored_ids.append(motor_id)
        return errored_ids

    def sync_write(
        self,
        motor_ids: Sequence[int],
        values: Sequence[Union[int, float]],
        address: int,
        size: int,
    ):
        self.check_connected()
        key = (address, size)
        if key not in self._sync_writers:
            self._sync_writers[key] = self.dxl.GroupSyncWrite(
                self.port_handler, self.packet_handler, address, size
            )
        sync_writer = self._sync_writers[key]

        errored_ids = []
        for motor_id, desired_pos in zip(motor_ids, values):
            value = signed_to_unsigned(int(desired_pos), size=size)
            value = value.to_bytes(size, byteorder="little")
            success = sync_writer.addParam(motor_id, value)
            if not success:
                errored_ids.append(motor_id)

        if errored_ids:
            logging.error("Sync write failed for: %s", str(errored_ids))

        comm_result = sync_writer.txPacket()
        self.handle_packet_result(comm_result, context="sync_write")

        sync_writer.clearParam()

    def check_connected(self):
        if self.lazy_connect and not self.is_connected:
            self.connect()
        if not self.is_connected:
            raise OSError("Must call connect() first.")

    def handle_packet_result(
        self,
        comm_result: int,
        dxl_error: Optional[int] = None,
        dxl_id: Optional[int] = None,
        context: Optional[str] = None,
    ):
        error_message = None
        if comm_result != self.dxl.COMM_SUCCESS:
            error_message = self.packet_handler.getTxRxResult(comm_result)
        elif dxl_error is not None:
            error_message = self.packet_handler.getRxPacketError(dxl_error)
        if error_message:
            if dxl_id is not None:
                error_message = "[Motor ID: {}] {}".format(dxl_id, error_message)
            if context is not None:
                error_message = "> {}: {}".format(context, error_message)
            logging.error(error_message)
            return False
        return True

    def convert_to_unsigned(self, value: int, size: int) -> int:
        if value < 0:
            max_value = (1 << (8 * size)) - 1
            value = max_value + value
        return value

    def __enter__(self):
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def __del__(self):
        self.disconnect()


class DynamixelReader:
    def __init__(
        self, client: DynamixelClient, motor_ids: Sequence[int], address: int, size: int
    ):
        self.client = client
        self.motor_ids = motor_ids
        self.address = address
        self.size = size
        self._initialize_data()

        self.operation = self.client.dxl.GroupBulkRead(
            client.port_handler, client.packet_handler
        )

        for motor_id in motor_ids:
            success = self.operation.addParam(motor_id, address, size)
            if not success:
                raise OSError(
                    "[Motor ID: {}] Could not add parameter to bulk read.".format(
                        motor_id
                    )
                )

    def read(self, retries: int = 1):
        self.client.check_connected()
        success = False
        while not success and retries >= 0:
            comm_result = self.operation.txRxPacket()
            success = self.client.handle_packet_result(comm_result, context="read")
            retries -= 1

        if not success:
            return self._get_data()

        errored_ids = []
        for i, motor_id in enumerate(self.motor_ids):
            available = self.operation.isAvailable(motor_id, self.address, self.size)
            if not available:
                errored_ids.append(motor_id)
                continue

            self._update_data(i, motor_id)

        if errored_ids:
            logging.error("Bulk read data is unavailable for: %s", str(errored_ids))

        return self._get_data()

    def _initialize_data(self):
        self._data = np.zeros(len(self.motor_ids), dtype=np.float32)

    def _update_data(self, index: int, motor_id: int):
        self._data[index] = self.operation.getData(motor_id, self.address, self.size)

    def _get_data(self):
        return self._data.copy()


class DynamixelPosVelCurReader(DynamixelReader):
    def __init__(
        self,
        client: DynamixelClient,
        motor_ids: Sequence[int],
        pos_scale: float = 1.0,
        vel_scale: float = 1.0,
        cur_scale: float = 1.0,
    ):
        super().__init__(
            client,
            motor_ids,
            address=ADDR_PRESENT_POS_VEL_CUR,
            size=LEN_PRESENT_POS_VEL_CUR,
        )
        self.pos_scale = pos_scale
        self.vel_scale = vel_scale
        self.cur_scale = cur_scale

    def _initialize_data(self):
        self._pos_data = np.zeros(len(self.motor_ids), dtype=np.float32)
        self._vel_data = np.zeros(len(self.motor_ids), dtype=np.float32)
        self._cur_data = np.zeros(len(self.motor_ids), dtype=np.float32)

    def _update_data(self, index: int, motor_id: int):
        cur = self.operation.getData(
            motor_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
        )
        vel = self.operation.getData(
            motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
        )
        pos = self.operation.getData(
            motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )
        cur = unsigned_to_signed(cur, size=2)
        vel = unsigned_to_signed(vel, size=4)
        pos = unsigned_to_signed(pos, size=4)
        self._pos_data[index] = float(pos) * self.pos_scale
        self._vel_data[index] = float(vel) * self.vel_scale
        self._cur_data[index] = float(cur) * self.cur_scale

    def _get_data(self):
        return (self._pos_data.copy(), self._vel_data.copy(), self._cur_data.copy())


class DynamixelPosReader(DynamixelReader):
    def __init__(
        self,
        client: DynamixelClient,
        motor_ids: Sequence[int],
        pos_scale: float = 1.0,
        vel_scale: float = 1.0,
        cur_scale: float = 1.0,
    ):
        super().__init__(
            client,
            motor_ids,
            address=ADDR_PRESENT_POS_VEL_CUR,
            size=LEN_PRESENT_POS_VEL_CUR,
        )
        self.pos_scale = pos_scale

    def _initialize_data(self):
        self._pos_data = np.zeros(len(self.motor_ids), dtype=np.float32)

    def _update_data(self, index: int, motor_id: int):
        pos = self.operation.getData(
            motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )
        pos = unsigned_to_signed(pos, size=4)
        self._pos_data[index] = float(pos) * self.pos_scale

    def _get_data(self):
        return self._pos_data.copy()


class DynamixelVelReader(DynamixelReader):
    def __init__(
        self,
        client: DynamixelClient,
        motor_ids: Sequence[int],
        pos_scale: float = 1.0,
        vel_scale: float = 1.0,
        cur_scale: float = 1.0,
    ):
        super().__init__(
            client,
            motor_ids,
            address=ADDR_PRESENT_POS_VEL_CUR,
            size=LEN_PRESENT_POS_VEL_CUR,
        )
        self.pos_scale = pos_scale
        self.vel_scale = vel_scale
        self.cur_scale = cur_scale

    def _initialize_data(self):
        self._vel_data = np.zeros(len(self.motor_ids), dtype=np.float32)

    def _update_data(self, index: int, motor_id: int):
        vel = self.operation.getData(
            motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
        )
        vel = unsigned_to_signed(vel, size=4)
        self._vel_data[index] = float(vel) * self.vel_scale

    def _get_data(self):
        return self._vel_data.copy()


class DynamixelCurReader(DynamixelReader):
    def __init__(
        self,
        client: DynamixelClient,
        motor_ids: Sequence[int],
        pos_scale: float = 1.0,
        vel_scale: float = 1.0,
        cur_scale: float = 1.0,
    ):
        super().__init__(
            client,
            motor_ids,
            address=ADDR_PRESENT_POS_VEL_CUR,
            size=LEN_PRESENT_POS_VEL_CUR,
        )
        self.cur_scale = cur_scale

    def _initialize_data(self):
        self._cur_data = np.zeros(len(self.motor_ids), dtype=np.float32)

    def _update_data(self, index: int, motor_id: int):
        cur = self.operation.getData(
            motor_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
        )
        cur = unsigned_to_signed(cur, size=2)
        self._cur_data[index] = float(cur) * self.cur_scale

    def _get_data(self):
        return self._cur_data.copy()


atexit.register(dynamixel_cleanup_handler)

if __name__ == "__main__":
    import argparse
    import itertools

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--motors", required=True, help="Comma-separated list of motor IDs."
    )
    parser.add_argument(
        "-d",
        "--device",
        default="/dev/ttyUSB0",
        help="The Dynamixel device to connect to.",
    )
    parser.add_argument(
        "-b", "--baud", default=1000000, help="The baudrate to connect with."
    )
    parsed_args = parser.parse_args()

    motors = [int(motor) for motor in parsed_args.motors.split(",")]

    way_points = [np.zeros(len(motors)), np.full(len(motors), np.pi)]

    with DynamixelClient(motors, parsed_args.device, parsed_args.baud) as dxl_client:
        for step in itertools.count():
            if step > 0 and step % 50 == 0:
                way_point = way_points[(step // 100) % len(way_points)]
                console.print("正在写入: {}".format(way_point.tolist()))
                dxl_client.write_desired_pos(motors, way_point)
            read_start = time.time()
            pos_now, vel_now, cur_now = dxl_client.read_pos_vel_cur()
            if step % 5 == 0:
                console.print(
                    "[{}] 频率: {:.2f} Hz".format(
                        step, 1.0 / (time.time() - read_start)
                    )
                )
                console.print("> 位置: {}".format(pos_now.tolist()))
                console.print("> 速度: {}".format(vel_now.tolist()))
                console.print("> 电流: {}".format(cur_now.tolist()))
