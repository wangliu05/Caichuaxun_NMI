import csv
import struct
import time
from pathlib import Path

import numpy as np


class MockActuator:
    def __init__(self, log_path=None, save_log=False):
        self.save_log = bool(save_log)
        self.log_path = Path(log_path) if log_path is not None else None
        self._header_written = False
        if self.save_log:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._header_written = self.log_path.exists() and self.log_path.stat().st_size > 0

    def send(self, q_rad, metadata=None):
        q_rad = np.asarray(q_rad, dtype=float).reshape(-1)
        servo_deg = None
        if metadata and "config" in metadata:
            servo_deg = model_q_to_servo_degrees(q_rad, metadata["config"])
        if not self.save_log:
            msg = f"[Actuator:mock] command received: q_axes={len(q_rad)}"
            if servo_deg is not None:
                msg += f", servo_axes={len(servo_deg)}"
            print(msg)
            return

        row = {
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        for i, value in enumerate(q_rad):
            row[f"q{i:02d}_rad"] = float(value)
        if servo_deg is not None:
            for i, value in enumerate(servo_deg):
                row[f"servo{i:02d}_deg"] = float(value)

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)
        print(f"[Actuator:mock] command logged: {self.log_path}")


class PCA9685Board:
    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, bus, address, frequency_hz=50.0):
        self.bus = bus
        self.address = int(address)
        self.frequency_hz = float(frequency_hz)
        self._init_board()

    def _init_board(self):
        self.bus.write_byte_data(self.address, self.MODE1, 0x00)
        time.sleep(0.005)
        self.set_pwm_frequency(self.frequency_hz)

    def set_pwm_frequency(self, frequency_hz):
        frequency_hz = float(frequency_hz)
        prescale_value = 25_000_000.0 / (4096.0 * frequency_hz) - 1.0
        prescale = int(round(prescale_value))

        old_mode = self.bus.read_byte_data(self.address, self.MODE1)
        sleep_mode = (old_mode & 0x7F) | 0x10
        self.bus.write_byte_data(self.address, self.MODE1, sleep_mode)
        self.bus.write_byte_data(self.address, self.PRESCALE, prescale)
        self.bus.write_byte_data(self.address, self.MODE1, old_mode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.MODE1, old_mode | 0xA1)

    def set_pwm(self, channel, on_tick, off_tick):
        channel = int(channel)
        if channel < 0 or channel > 15:
            raise ValueError(f"PCA9685 channel out of range: {channel}")
        base = self.LED0_ON_L + 4 * channel
        values = [
            int(on_tick) & 0xFF,
            (int(on_tick) >> 8) & 0x0F,
            int(off_tick) & 0xFF,
            (int(off_tick) >> 8) & 0x0F,
        ]
        self.bus.write_i2c_block_data(self.address, base, values)


class PCA9685Actuator:
    def __init__(self, config):
        try:
            from smbus2 import SMBus
        except ImportError as exc:
            raise RuntimeError(
                "PCA9685 actuator backend requires smbus2. Install it with: pip install smbus2"
            ) from exc

        self.config = config
        self.bus = SMBus(int(config.PCA_I2C_BUS))
        self.addresses = tuple(int(a) for a in config.PCA_I2C_ADDRESSES)
        self.boards = [
            PCA9685Board(self.bus, address, config.PCA_PWM_FREQUENCY_HZ)
            for address in self.addresses
        ]
        self.channel_map = build_channel_map(config, len(self.boards))
        print(
            f"[Actuator:pca9685] bus={config.PCA_I2C_BUS} "
            f"addresses={[hex(a) for a in self.addresses]} axes={len(self.channel_map)}"
        )

    def send(self, q_rad, metadata=None):
        del metadata
        servo_deg = model_q_to_servo_degrees(q_rad, self.config)
        if len(servo_deg) > len(self.channel_map):
            raise ValueError(f"Got {len(servo_deg)} servo axes but channel map has {len(self.channel_map)} entries")

        for axis_index, angle_deg in enumerate(servo_deg):
            board_index, channel = self.channel_map[axis_index]
            pulse_us = servo_angle_deg_to_pulse_us(angle_deg, axis_index, self.config)
            off_tick = pulse_us_to_ticks(pulse_us, self.config.PCA_PWM_FREQUENCY_HZ)
            self.boards[int(board_index)].set_pwm(int(channel), 0, off_tick)
        print(f"[Actuator:pca9685] sent {len(servo_deg)} PWM channels")

    def set_servo_degrees(self, servo_deg):
        servo_deg = np.asarray(servo_deg, dtype=float).reshape(-1)
        if len(servo_deg) > len(self.channel_map):
            raise ValueError(f"Got {len(servo_deg)} servo axes but channel map has {len(self.channel_map)} entries")
        for axis_index, angle_deg in enumerate(servo_deg):
            self.set_axis_degrees(axis_index, angle_deg)

    def set_axis_degrees(self, axis_index, angle_deg, pulse_offset_us=0.0):
        board_index, channel = self.channel_map[int(axis_index)]
        pulse_us = servo_angle_deg_to_pulse_us(angle_deg, int(axis_index), self.config) + float(pulse_offset_us)
        pulse_us = np.clip(pulse_us, self.config.SERVO_MIN_PULSE_US, self.config.SERVO_MAX_PULSE_US)
        off_tick = pulse_us_to_ticks(pulse_us, self.config.PCA_PWM_FREQUENCY_HZ)
        self.boards[int(board_index)].set_pwm(int(channel), 0, off_tick)

    def close(self):
        if self.bus is not None:
            self.bus.close()
            self.bus = None


def build_channel_map(config, board_count):
    if config.PCA_CHANNEL_MAP is not None:
        return tuple((int(b), int(c)) for b, c in config.PCA_CHANNEL_MAP)

    mapping = []
    for axis in range(36):
        board_index = axis // 16
        channel = axis % 16
        if board_index >= board_count:
            raise ValueError("Default 36-axis map requires at least three 16-channel PCA9685 boards")
        mapping.append((board_index, channel))
    return tuple(mapping)


class PIDController:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limit = abs(float(output_limit))
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, dt):
        dt = max(float(dt), 1e-6)
        self.integral += float(error) * dt
        derivative = 0.0 if self.prev_error is None else (float(error) - self.prev_error) / dt
        self.prev_error = float(error)
        out = self.kp * float(error) + self.ki * self.integral + self.kd * derivative
        return float(np.clip(out, -self.output_limit, self.output_limit))


class EncoderReader:
    def read_degrees(self, axis_index):
        raise NotImplementedError

    def close(self):
        pass


class NoEncoderReader(EncoderReader):
    def read_degrees(self, axis_index):
        raise RuntimeError(
            "Encoder PID is enabled, but ENCODER_BACKEND='none'. "
            "Provide the grating-encoder readout protocol before using closed-loop PID."
        )


class ModbusRtuClient:
    def __init__(self, port, baudrate=9600, bytesize=8, parity="N", stopbits=1, timeout=0.08):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("FT-EPC-04 encoder backend requires pyserial. Install it with: pip install pyserial") from exc

        self.serial = serial.Serial(
            port=port,
            baudrate=int(baudrate),
            bytesize=int(bytesize),
            parity=str(parity),
            stopbits=int(stopbits),
            timeout=float(timeout),
        )

    def read_holding_registers(self, slave_id, start_register, count):
        request = bytes([
            int(slave_id) & 0xFF,
            0x03,
            (int(start_register) >> 8) & 0xFF,
            int(start_register) & 0xFF,
            (int(count) >> 8) & 0xFF,
            int(count) & 0xFF,
        ])
        request += modbus_crc16_bytes(request)
        self.serial.reset_input_buffer()
        self.serial.write(request)
        expected_len = 5 + 2 * int(count)
        response = self.serial.read(expected_len)
        if len(response) != expected_len:
            raise RuntimeError(f"Modbus timeout/short response: got {len(response)} bytes, expected {expected_len}")
        check_modbus_response(response, int(slave_id), 0x03)
        byte_count = response[2]
        if byte_count != 2 * int(count):
            raise RuntimeError(f"Unexpected Modbus byte count: {byte_count}")
        data = response[3:3 + byte_count]
        return [int.from_bytes(data[i:i + 2], byteorder="big", signed=False) for i in range(0, len(data), 2)]

    def close(self):
        self.serial.close()


class FTEPC04EncoderReader(EncoderReader):
    """
    FT-EPC-04 grating/incremental encoder module reader.

    Manual V4.2:
    - RS485 Modbus-RTU, default 9600 N81, default slave id 1.
    - Function code 0x03 reads encoder counts.
    - CH1..CH4 counts are signed 32-bit values at registers:
      0x0030-0x0031, 0x0032-0x0033, 0x0034-0x0035, 0x0036-0x0037.
    """

    def __init__(self, config):
        self.config = config
        self.client = ModbusRtuClient(
            config.ENCODER_SERIAL_PORT,
            baudrate=config.ENCODER_BAUDRATE,
            bytesize=config.ENCODER_BYTESIZE,
            parity=config.ENCODER_PARITY,
            stopbits=config.ENCODER_STOPBITS,
            timeout=config.ENCODER_TIMEOUT_S,
        )
        self.axis_map = build_encoder_axis_map(config)
        self.cache = {}
        self.cache_time = {}
        print(
            f"[Encoder:ft-epc-04] port={config.ENCODER_SERIAL_PORT} "
            f"baud={config.ENCODER_BAUDRATE} axes={len(self.axis_map)}"
        )

    def read_degrees(self, axis_index):
        count = self.read_count(axis_index)
        return encoder_count_to_alpha_deg(count, axis_index, self.config)

    def read_count(self, axis_index):
        slave_id, channel = self.axis_map[int(axis_index)]
        counts = self._read_module_counts(int(slave_id))
        return counts[int(channel)]

    def _read_module_counts(self, slave_id):
        now = time.monotonic()
        ttl = float(self.config.ENCODER_CACHE_TTL_S)
        if slave_id in self.cache and now - self.cache_time.get(slave_id, 0.0) <= ttl:
            return self.cache[slave_id]

        regs = self.client.read_holding_registers(
            slave_id,
            int(self.config.ENCODER_COUNT_START_REGISTER),
            int(self.config.ENCODER_COUNT_REGISTER_COUNT),
        )
        counts = []
        for i in range(0, 8, 2):
            counts.append(register_pair_to_int32(regs[i], regs[i + 1], self.config.ENCODER_WORD_ORDER))
        self.cache[slave_id] = counts
        self.cache_time[slave_id] = now
        return counts

    def close(self):
        self.client.close()


def create_encoder_reader(config):
    if config.ENCODER_BACKEND in ("none", None):
        return NoEncoderReader()
    if config.ENCODER_BACKEND in ("ft_epc_04", "ftepc04", "ft-epc-04", "modbus_rtu"):
        return FTEPC04EncoderReader(config)
    raise NotImplementedError(
        f"Unsupported ENCODER_BACKEND={config.ENCODER_BACKEND}. "
        "Supported: ft_epc_04"
    )


class EncoderPidPCA9685Actuator:
    """
    Closed-loop wrapper for PCA9685 PWM output plus external grating encoder feedback.

    PCA9685 does not read encoders. This class requires a separate EncoderReader
    backend that returns actual axis angles in degrees.
    """

    def __init__(self, pca_actuator, encoder_reader, config):
        self.pca = pca_actuator
        self.encoder = encoder_reader
        self.config = config

    def send(self, q_rad, metadata=None):
        del metadata
        targets = model_q_to_servo_degrees(q_rad, self.config)
        self.pca.set_servo_degrees(targets)

        deadline = time.monotonic() + float(self.config.ENCODER_PID_TIMEOUT_S)
        dt = float(self.config.ENCODER_PID_DT_S)
        controllers = [
            PIDController(
                self.config.ENCODER_PID_KP,
                self.config.ENCODER_PID_KI,
                self.config.ENCODER_PID_KD,
                self.config.ENCODER_PWM_CORRECTION_LIMIT_US,
            )
            for _ in range(len(targets))
        ]

        active = set(range(len(targets)))
        while active and time.monotonic() < deadline:
            for axis in list(active):
                measured = self.encoder.read_degrees(axis)
                error = wrapped_angle_error_deg(targets[axis], measured)
                if abs(error) <= float(self.config.ENCODER_PID_TOL_DEG):
                    active.remove(axis)
                    continue
                correction_us = controllers[axis].update(error, dt)
                self.pca.set_axis_degrees(axis, targets[axis], pulse_offset_us=correction_us)
            time.sleep(dt)

        if active:
            print(f"[Actuator:encoder-pid] warning: axes not settled: {sorted(active)}")
        else:
            print("[Actuator:encoder-pid] all axes settled")

    def close(self):
        self.encoder.close()
        self.pca.close()


def wrapped_angle_error_deg(target_deg, measured_deg):
    return (float(target_deg) - float(measured_deg) + 180.0) % 360.0 - 180.0


def build_encoder_axis_map(config):
    if config.ENCODER_AXIS_MAP is not None:
        return tuple((int(slave), int(ch)) for slave, ch in config.ENCODER_AXIS_MAP)

    slave_ids = tuple(int(v) for v in config.ENCODER_SLAVE_IDS)
    mapping = []
    for axis in range(36):
        module_index = axis // 4
        channel = axis % 4
        if module_index >= len(slave_ids):
            raise ValueError("Default 36-axis encoder map requires nine FT-EPC-04 modules")
        mapping.append((slave_ids[module_index], channel))
    return tuple(mapping)


def encoder_count_to_alpha_deg(count, axis_index, config):
    zero_counts = np.asarray(config.ENCODER_ZERO_COUNTS, dtype=float)
    zero_alpha = np.asarray(config.ENCODER_ZERO_ALPHA_DEG, dtype=float)
    direction = np.asarray(config.ENCODER_ALPHA_DIR, dtype=float)
    cpr = float(config.ENCODER_COUNTS_PER_REV)
    if cpr <= 0:
        raise ValueError("ENCODER_COUNTS_PER_REV must be positive")
    axis = int(axis_index)
    alpha = zero_alpha[axis] + direction[axis] * (float(count) - zero_counts[axis]) * 360.0 / cpr
    return float(np.mod(alpha, 360.0))


def register_pair_to_int32(reg0, reg1, word_order="high_low"):
    if str(word_order).lower() in ("low_high", "little", "word_swap"):
        raw = int(reg1).to_bytes(2, "big") + int(reg0).to_bytes(2, "big")
    else:
        raw = int(reg0).to_bytes(2, "big") + int(reg1).to_bytes(2, "big")
    return struct.unpack(">i", raw)[0]


def modbus_crc16(data):
    crc = 0xFFFF
    for b in data:
        crc ^= int(b)
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def modbus_crc16_bytes(data):
    crc = modbus_crc16(data)
    return bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def check_modbus_response(response, slave_id, function_code):
    if response[0] != int(slave_id):
        raise RuntimeError(f"Unexpected Modbus slave id: {response[0]} != {slave_id}")
    if response[1] == (function_code | 0x80):
        raise RuntimeError(f"Modbus exception code: {response[2]}")
    if response[1] != int(function_code):
        raise RuntimeError(f"Unexpected Modbus function: {response[1]} != {function_code}")
    payload, received_crc = response[:-2], response[-2:]
    expected_crc = modbus_crc16_bytes(payload)
    if received_crc != expected_crc:
        raise RuntimeError(f"Bad Modbus CRC: got {received_crc.hex()}, expected {expected_crc.hex()}")


def model_q_to_servo_degrees(q_rad, config):
    q = np.asarray(q_rad, dtype=float).reshape(-1)
    if len(q) % 2 != 0:
        raise ValueError("Model q must contain [gamma,beta] pairs")

    gamma_init = np.asarray(config.SERVO_GAMMA_INIT_DEG, dtype=float)
    beta_init = np.asarray(config.SERVO_BETA_INIT_DEG, dtype=float)
    gamma_dir = np.asarray(config.SERVO_GAMMA_DIR, dtype=float)
    beta_dir = np.asarray(config.SERVO_BETA_DIR, dtype=float)
    magnet_count = len(q) // 2
    if len(gamma_init) < magnet_count or len(beta_init) < magnet_count:
        raise ValueError("Servo initial angle arrays are shorter than magnet count")

    q_pairs = q.reshape(magnet_count, 2)
    servo = np.empty(magnet_count * 2, dtype=float)
    for i, (gamma, beta) in enumerate(q_pairs):
        servo[2 * i] = gamma_init[i] + gamma_dir[i] * np.rad2deg(gamma)
        servo[2 * i + 1] = beta_init[i] + beta_dir[i] * np.rad2deg(beta)
    return np.mod(servo, 360.0)


def servo_angle_deg_to_pulse_us(angle_deg, axis_index, config):
    a0 = float(config.SERVO_MIN_ANGLE_DEG)
    a1 = float(config.SERVO_MAX_ANGLE_DEG)
    if abs(a1 - a0) < 1e-12:
        raise ValueError("SERVO_MIN_ANGLE_DEG and SERVO_MAX_ANGLE_DEG must differ")

    angle_raw = float(angle_deg)
    if a0 <= angle_raw <= a1:
        angle = angle_raw
    else:
        span = a1 - a0
        angle = a0 + np.mod(angle_raw - a0, span)

    if axis_index in set(int(i) for i in config.SERVO_INVERTED_AXES):
        angle = a1 - (angle - a0)

    ratio = (angle - a0) / (a1 - a0)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    p0 = float(config.SERVO_MIN_PULSE_US)
    p1 = float(config.SERVO_MAX_PULSE_US)
    return p0 + ratio * (p1 - p0)


def pulse_us_to_ticks(pulse_us, frequency_hz):
    period_us = 1_000_000.0 / float(frequency_hz)
    ticks = int(round(float(pulse_us) / period_us * 4096.0))
    return max(0, min(4095, ticks))


def create_actuator(config):
    if config.ACTUATOR_BACKEND == "mock":
        return MockActuator(config.ACTUATOR_LOG, getattr(config, "ACTUATOR_SAVE_LOG", False))
    if config.ACTUATOR_BACKEND in ("pca9685", "pca9585"):
        pca = PCA9685Actuator(config)
        if getattr(config, "ACTUATOR_USE_ENCODER_PID", False):
            return EncoderPidPCA9685Actuator(pca, create_encoder_reader(config), config)
        return pca
    raise NotImplementedError(f"Unsupported actuator backend: {config.ACTUATOR_BACKEND}")
