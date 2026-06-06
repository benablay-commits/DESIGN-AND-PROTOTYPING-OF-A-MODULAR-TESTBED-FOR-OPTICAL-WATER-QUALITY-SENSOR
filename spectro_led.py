import collections
import csv
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PyQt6 import QtCore, QtGui, QtWidgets

APP_INSTANCE = None


WAVELENGTHS = np.array(
    [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940],
    dtype=float,
)

CHANNEL_LABELS = [f"{int(wl)} nm" for wl in WAVELENGTHS]
CHANNEL_COLORS = [
    "#7f00ff", "#4a00ff", "#005bff", "#00b7ff", "#00ffd5", "#00ff91",
    "#48ff00", "#adff00", "#ffe100", "#ffb000", "#ff7a00", "#ff4d00",
    "#ff0037", "#ff006e", "#ff00c8", "#d400ff", "#9b00ff", "#6a00ff",
]
VIBRATOR_SUPPLY_VOLTAGE = 5.0


def spectral_gradient_image(width: int = 900, height: int = 300) -> np.ndarray:
    image = np.zeros((height, width, 4), dtype=np.ubyte)
    stops = [
        (0.00, QtGui.QColor(20, 70, 255)),
        (0.22, QtGui.QColor(0, 190, 255)),
        (0.42, QtGui.QColor(0, 220, 90)),
        (0.60, QtGui.QColor(255, 230, 0)),
        (0.78, QtGui.QColor(255, 120, 0)),
        (1.00, QtGui.QColor(230, 20, 15)),
    ]

    for column in range(width):
        fraction = column / max(1, width - 1)
        left_stop, right_stop = stops[0], stops[-1]
        for current, next_stop in zip(stops[:-1], stops[1:]):
            if current[0] <= fraction <= next_stop[0]:
                left_stop, right_stop = current, next_stop
                break

        span = max(0.0001, right_stop[0] - left_stop[0])
        local_t = (fraction - left_stop[0]) / span
        left_color = left_stop[1]
        right_color = right_stop[1]
        color = QtGui.QColor(
            int(left_color.red() + (right_color.red() - left_color.red()) * local_t),
            int(left_color.green() + (right_color.green() - left_color.green()) * local_t),
            int(left_color.blue() + (right_color.blue() - left_color.blue()) * local_t),
        )
        image[:, column, 0] = color.red()
        image[:, column, 1] = color.green()
        image[:, column, 2] = color.blue()
        image[:, column, 3] = 225
    return image


class SpectrumAreaItem(pg.GraphicsObject):
    def __init__(self, wavelengths: np.ndarray) -> None:
        super().__init__()
        self.wavelengths = np.array(wavelengths, dtype=float)
        self.values = np.full(len(wavelengths), np.nan, dtype=float)
        self.picture = QtGui.QPicture()
        self._bounds = QtCore.QRectF(float(self.wavelengths[0]), 0.0, float(self.wavelengths[-1] - self.wavelengths[0]), 1.0)
        self._generate_picture()

    def setData(self, values: np.ndarray) -> None:
        self.values = np.array(values, dtype=float)
        self._generate_picture()
        self.update()

    def paint(self, painter, option, widget=None) -> None:
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self) -> QtCore.QRectF:
        return self._bounds

    def _generate_picture(self) -> None:
        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        finite_values = np.nan_to_num(self.values, nan=0.0)
        max_value = max(1.0, float(np.max(finite_values)) if finite_values.size else 1.0)
        self._bounds = QtCore.QRectF(
            float(self.wavelengths[0]),
            0.0,
            float(self.wavelengths[-1] - self.wavelengths[0]),
            max_value * 1.08,
        )

        for left_wl, right_wl, left_value, right_value in zip(
            self.wavelengths[:-1],
            self.wavelengths[1:],
            finite_values[:-1],
            finite_values[1:],
        ):
            color = wavelength_to_rgb(float((left_wl + right_wl) / 2.0), 1.0)
            color.setAlpha(210)
            path = QtGui.QPainterPath(QtCore.QPointF(float(left_wl), 0.0))
            path.lineTo(float(left_wl), float(left_value))
            path.lineTo(float(right_wl), float(right_value))
            path.lineTo(float(right_wl), 0.0)
            path.closeSubpath()
            painter.fillPath(path, QtGui.QBrush(color))

        painter.end()
        self.picture = picture


class SpectrumGridItem(pg.GraphicsObject):
    def __init__(self, x_values, y_steps: int = 5) -> None:
        super().__init__()
        self.x_values = [float(value) for value in x_values]
        self.y_max = 1.0
        self.y_steps = y_steps
        self.picture = QtGui.QPicture()
        self._bounds = QtCore.QRectF(float(WAVELENGTHS[0]), 0.0, float(WAVELENGTHS[-1] - WAVELENGTHS[0]), 1.0)
        self._generate_picture()

    def setYMax(self, y_max: float) -> None:
        self.y_max = max(1.0, float(y_max))
        self._bounds = QtCore.QRectF(float(WAVELENGTHS[0]), 0.0, float(WAVELENGTHS[-1] - WAVELENGTHS[0]), self.y_max)
        self._generate_picture()
        self.update()

    def paint(self, painter, option, widget=None) -> None:
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self) -> QtCore.QRectF:
        return self._bounds

    def _generate_picture(self) -> None:
        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        pen = QtGui.QPen(QtGui.QColor(50, 160, 80, 190), 0)
        painter.setPen(pen)

        for x_value in self.x_values:
            painter.drawLine(QtCore.QPointF(x_value, 0.0), QtCore.QPointF(x_value, self.y_max))

        for step in range(1, self.y_steps):
            y_value = self.y_max * step / self.y_steps
            painter.drawLine(
                QtCore.QPointF(float(WAVELENGTHS[0]), y_value),
                QtCore.QPointF(float(WAVELENGTHS[-1]), y_value),
            )

        painter.end()
        self.picture = picture


class SpectrumChannelBandsItem(pg.GraphicsObject):
    def __init__(self, wavelengths: np.ndarray) -> None:
        super().__init__()
        self.wavelengths = np.array(wavelengths, dtype=float)
        self.y_max = 1.0
        self.picture = QtGui.QPicture()
        self._bounds = QtCore.QRectF(float(self.wavelengths[0]), 0.0, float(self.wavelengths[-1] - self.wavelengths[0]), 1.0)
        self._generate_picture()

    def setYMax(self, y_max: float) -> None:
        self.y_max = max(1.0, float(y_max))
        self._bounds = QtCore.QRectF(float(self.wavelengths[0]), 0.0, float(self.wavelengths[-1] - self.wavelengths[0]), self.y_max)
        self._generate_picture()
        self.update()

    def paint(self, painter, option, widget=None) -> None:
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self) -> QtCore.QRectF:
        return self._bounds

    def _channel_edges(self) -> list[float]:
        edges = [float(self.wavelengths[0])]
        for left, right in zip(self.wavelengths[:-1], self.wavelengths[1:]):
            edges.append(float((left + right) / 2.0))
        edges.append(float(self.wavelengths[-1]))
        return edges

    def _generate_picture(self) -> None:
        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)

        edges = self._channel_edges()
        for index, wavelength in enumerate(self.wavelengths):
            color = wavelength_to_rgb(float(wavelength), 0.95)
            color.setAlpha(150)
            rect = QtCore.QRectF(edges[index], 0.0, edges[index + 1] - edges[index], self.y_max)
            painter.fillRect(rect, color)

        painter.end()
        self.picture = picture


def wavelength_to_rgb(wavelength_nm: float, intensity: float = 1.0) -> QtGui.QColor:
    wavelength = float(wavelength_nm)
    intensity = max(0.0, min(1.0, float(intensity)))

    # Visible spectrum reference matching the provided diagram:
    # 740-625 red, 625-590 orange, 590-565 yellow,
    # 565-520 green, 520-500 cyan, 500-435 blue, 435-380 violet.
    if wavelength >= 625.0:
        base = QtGui.QColor(255, 40, 20)
    elif wavelength >= 590.0:
        t = (625.0 - wavelength) / (625.0 - 590.0)
        base = QtGui.QColor(
            int(255),
            int(120 + 80 * t),
            int(20 * (1.0 - t)),
        )
    elif wavelength >= 565.0:
        t = (590.0 - wavelength) / (590.0 - 565.0)
        base = QtGui.QColor(
            255,
            220 + int(35 * t),
            0,
        )
    elif wavelength >= 520.0:
        t = (565.0 - wavelength) / (565.0 - 520.0)
        base = QtGui.QColor(
            int(120 * (1.0 - t)),
            235,
            int(30 * t),
        )
    elif wavelength >= 500.0:
        t = (520.0 - wavelength) / (520.0 - 500.0)
        base = QtGui.QColor(
            0,
            int(220 + 20 * (1.0 - t)),
            int(160 + 70 * t),
        )
    elif wavelength >= 435.0:
        t = (500.0 - wavelength) / (500.0 - 435.0)
        base = QtGui.QColor(
            int(10 + 40 * t),
            int(170 * (1.0 - t)),
            255,
        )
    elif wavelength >= 380.0:
        t = (435.0 - wavelength) / (435.0 - 380.0)
        base = QtGui.QColor(
            int(120 + 50 * t),
            int(20 + 10 * t),
            int(255 - 30 * t),
        )
    elif wavelength > 0.0:
        # UV: show as dim violet because it is not visible to the human eye.
        base = QtGui.QColor(110, 40, 180)
    else:
        base = QtGui.QColor(0, 0, 0)

    if wavelength > 740.0:
        # NIR: not visible, use dim deep red.
        fade = max(0.18, 1.0 - min(1.0, (wavelength - 740.0) / 200.0))
        base = QtGui.QColor(int(150 * fade), 0, 0)

    brightness = 0.18 + 0.82 * intensity
    return QtGui.QColor(
        min(255, int(base.red() * brightness)),
        min(255, int(base.green() * brightness)),
        min(255, int(base.blue() * brightness)),
    )


class SpectralColorStrip(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.values = np.zeros(len(WAVELENGTHS), dtype=float)
        self.setMinimumHeight(170)
        self.setMaximumHeight(190)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def update_values(self, values: np.ndarray) -> None:
        self.values = np.array(values, dtype=float)
        self.update()

    def _x_for_wavelength(self, wavelength: float, left: int, width: int) -> int:
        sensor_min = float(WAVELENGTHS[0])
        sensor_max = float(WAVELENGTHS[-1])
        clamped = max(sensor_min, min(sensor_max, wavelength))
        fraction = (clamped - sensor_min) / (sensor_max - sensor_min)
        return left + int(width * fraction)

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor(8, 15, 30))

        left_margin = 28
        right_margin = 28
        top_margin = 12
        spectrum_height = 62
        label_top = top_margin + spectrum_height + 12
        spectrum_rect = QtCore.QRect(left_margin, top_margin, rect.width() - left_margin - right_margin, spectrum_height)

        gradient = QtGui.QLinearGradient(spectrum_rect.left(), 0, spectrum_rect.right(), 0)
        sensor_min = float(WAVELENGTHS[0])
        sensor_max = float(WAVELENGTHS[-1])

        def stop_for(wavelength: float) -> float:
            return (wavelength - sensor_min) / (sensor_max - sensor_min)

        gradient.setColorAt(0.00, QtGui.QColor(52, 22, 82))
        gradient.setColorAt(stop_for(435.0), QtGui.QColor(0, 22, 96))
        gradient.setColorAt(stop_for(500.0), QtGui.QColor(0, 78, 104))
        gradient.setColorAt(stop_for(520.0), QtGui.QColor(0, 80, 60))
        gradient.setColorAt(stop_for(565.0), QtGui.QColor(42, 92, 8))
        gradient.setColorAt(stop_for(590.0), QtGui.QColor(112, 96, 0))
        gradient.setColorAt(stop_for(625.0), QtGui.QColor(112, 52, 0))
        gradient.setColorAt(stop_for(740.0), QtGui.QColor(110, 16, 8))
        gradient.setColorAt(1.00, QtGui.QColor(22, 3, 3))

        painter.fillRect(spectrum_rect, gradient)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 70), 1))
        painter.drawRect(spectrum_rect)

        max_value = float(np.max(self.values)) if self.values.size else 0.0
        if max_value <= 0.0:
            max_value = 1.0

        count = len(WAVELENGTHS)
        band_width = spectrum_rect.width() / max(1, count)

        for index, wavelength in enumerate(WAVELENGTHS):
            intensity = float(self.values[index]) / max_value if index < len(self.values) else 0.0
            intensity = max(0.0, min(1.0, intensity))
            color = wavelength_to_rgb(float(wavelength), intensity)

            center_x = self._x_for_wavelength(float(wavelength), spectrum_rect.left(), spectrum_rect.width())
            glow_width = max(14, int(band_width * (0.75 + 1.2 * intensity)))
            glow_rect = QtCore.QRect(center_x - glow_width // 2, spectrum_rect.top(), glow_width, spectrum_rect.height())

            glow = QtGui.QLinearGradient(glow_rect.left(), 0, glow_rect.right(), 0)
            alpha = int(45 + 165 * intensity)
            glow.setColorAt(0.0, QtGui.QColor(color.red(), color.green(), color.blue(), 0))
            glow.setColorAt(0.5, QtGui.QColor(color.red(), color.green(), color.blue(), alpha))
            glow.setColorAt(1.0, QtGui.QColor(color.red(), color.green(), color.blue(), 0))
            painter.fillRect(glow_rect.intersected(spectrum_rect), glow)

            painter.setPen(QtGui.QColor(226, 232, 240))
            label_rect = QtCore.QRect(center_x - 22, label_top + (12 if index % 2 else 0), 44, 18)
            painter.drawText(label_rect, int(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop), str(int(wavelength)))

        painter.setPen(QtGui.QColor(148, 163, 184))
        painter.drawText(left_margin, rect.height() - 14, 'Detected channel glow across AS7265x range (410 nm -> 940 nm)')

@dataclass
class SensorFrame:
    timestamp_ms: int
    values: np.ndarray


@dataclass
class TemperatureFrame:
    timestamp_ms: int
    temperature_c: float


class SerialReader(QtCore.QObject):
    frame_received = QtCore.pyqtSignal(object)
    raw_frame_received = QtCore.pyqtSignal(object)
    temperature_received = QtCore.pyqtSignal(object)
    message_received = QtCore.pyqtSignal(str)
    connection_changed = QtCore.pyqtSignal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        self.serial_port = None
        self.running = False
        self.read_thread = None
        self.write_lock = threading.Lock()

    def connect_port(self, port_name: str, baudrate: int) -> None:
        self.disconnect_port()

        self.serial_port = serial.Serial(
            port=port_name,
            baudrate=baudrate,
            timeout=0.1,
            write_timeout=0.2,
            rtscts=False,
            dsrdtr=False,
        )
        self.running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        self.connection_changed.emit(True, f"Connected to {port_name} @ {baudrate}")

    def disconnect_port(self) -> None:
        self.running = False
        if self.serial_port is not None:
            try:
                if self.serial_port.is_open:
                    self.serial_port.close()
            except serial.SerialException:
                pass
        self.serial_port = None
        self.connection_changed.emit(False, "Not connected")

    def send_command(self, command: str) -> None:
        if self.serial_port is None or not self.serial_port.is_open:
            raise serial.SerialException("Not connected")

        with self.write_lock:
            self.serial_port.write((command + "\n").encode("utf-8"))
            self.serial_port.flush()

    def hard_reset(self) -> None:
        if self.serial_port is None or not self.serial_port.is_open:
            raise serial.SerialException("Not connected")

        with self.write_lock:
            self.serial_port.setDTR(False)
            self.serial_port.setRTS(True)
            time.sleep(0.12)
            self.serial_port.setRTS(False)
            self.serial_port.reset_input_buffer()

    def _read_loop(self) -> None:
        while self.running and self.serial_port is not None and self.serial_port.is_open:
            try:
                raw_line = self.serial_port.readline()
            except serial.SerialException as exc:
                self.message_received.emit(f"Serial error: {exc}")
                break

            if not raw_line:
                continue

            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            if line.startswith("DATA,"):
                frame = self._parse_frame(line)
                if frame is not None:
                    self.frame_received.emit(frame)
            elif line.startswith("RAW,"):
                frame = self._parse_frame(line)
                if frame is not None:
                    self.raw_frame_received.emit(frame)
            elif line.startswith("TEMP,"):
                temperature_frame = self._parse_temperature_frame(line)
                if temperature_frame is not None:
                    self.temperature_received.emit(temperature_frame)
            else:
                self.message_received.emit(line)

        self.connection_changed.emit(False, "Not connected")

    def _parse_frame(self, line: str):
        parts = line.split(",")
        if len(parts) != 20:
            self.message_received.emit(f"Invalid frame length: {line}")
            return None

        try:
            timestamp_ms = int(parts[1])
            values = np.array([float(item) for item in parts[2:]], dtype=float)
        except ValueError:
            self.message_received.emit(f"Invalid sensor frame: {line}")
            return None

        return SensorFrame(timestamp_ms=timestamp_ms, values=values)

    def _parse_temperature_frame(self, line: str):
        parts = line.split(",")
        if len(parts) != 3:
            self.message_received.emit(f"Invalid temperature frame: {line}")
            return None

        try:
            timestamp_ms = int(parts[1])
            temperature_c = float(parts[2])
        except ValueError:
            self.message_received.emit(f"Invalid temperature value: {line}")
            return None

        return TemperatureFrame(timestamp_ms=timestamp_ms, temperature_c=temperature_c)


class LedButtonRow(QtWidgets.QWidget):
    command_requested = QtCore.pyqtSignal(str)

    def __init__(self, label: str, command_base: str) -> None:
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel(label))

        for title, suffix in [("On", "ON"), ("Off", "OFF"), ("Only", "ONLY")]:
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(lambda _checked=False, s=suffix: self.command_requested.emit(f"{command_base} {s}"))
            layout.addWidget(button)

        layout.addStretch(1)


class PeltierControlWindow(QtWidgets.QWidget):
    command_requested = QtCore.pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Standalone Peltier duty-cycle test")
        self.resize(620, 520)
        self.cycle_running = False
        self.cycle_mode = "HEAT"
        self.cycle_phase = "OFF"
        self.last_direction = None
        self.phase_timer = QtCore.QTimer(self)
        self.phase_timer.setSingleShot(True)
        self.phase_timer.timeout.connect(self._advance_cycle_phase)

        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                color: #111827;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #111827;
                margin-top: 10px;
                padding: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                background-color: white;
            }
            QPushButton {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #111827;
                padding: 8px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #111827;
                padding: 5px;
            }
            QPlainTextEdit {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #111827;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        settings_box = QtWidgets.QGroupBox("Peltier settings")
        settings_layout = QtWidgets.QGridLayout(settings_box)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Heat", "HEAT")
        self.mode_combo.addItem("Cool", "COOL")
        settings_layout.addWidget(QtWidgets.QLabel("Direction"), 0, 0)
        settings_layout.addWidget(self.mode_combo, 0, 1)

        self.pwm_spin = QtWidgets.QSpinBox()
        self.pwm_spin.setRange(0, 255)
        self.pwm_spin.setValue(172)
        self.pwm_spin.valueChanged.connect(self.update_preview)
        settings_layout.addWidget(QtWidgets.QLabel("PWM"), 1, 0)
        settings_layout.addWidget(self.pwm_spin, 1, 1)

        self.on_time_spin = QtWidgets.QDoubleSpinBox()
        self.on_time_spin.setRange(0.1, 600.0)
        self.on_time_spin.setDecimals(1)
        self.on_time_spin.setSingleStep(0.5)
        self.on_time_spin.setValue(5.0)
        self.on_time_spin.setSuffix(" s")
        self.on_time_spin.valueChanged.connect(self.update_preview)
        settings_layout.addWidget(QtWidgets.QLabel("On time"), 2, 0)
        settings_layout.addWidget(self.on_time_spin, 2, 1)

        self.off_time_spin = QtWidgets.QDoubleSpinBox()
        self.off_time_spin.setRange(0.1, 600.0)
        self.off_time_spin.setDecimals(1)
        self.off_time_spin.setSingleStep(0.5)
        self.off_time_spin.setValue(5.0)
        self.off_time_spin.setSuffix(" s")
        self.off_time_spin.valueChanged.connect(self.update_preview)
        settings_layout.addWidget(QtWidgets.QLabel("Total off time"), 3, 0)
        settings_layout.addWidget(self.off_time_spin, 3, 1)

        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setWordWrap(True)
        settings_layout.addWidget(self.preview_label, 4, 0, 1, 2)
        layout.addWidget(settings_box)

        control_box = QtWidgets.QGroupBox("Standalone test controls")
        control_layout = QtWidgets.QGridLayout(control_box)

        continuous_button = QtWidgets.QPushButton("Start continuous")
        continuous_button.clicked.connect(self.start_continuous)
        control_layout.addWidget(continuous_button, 0, 0)

        duty_button = QtWidgets.QPushButton("Start duty cycle")
        duty_button.clicked.connect(self.start_duty_cycle)
        control_layout.addWidget(duty_button, 0, 1)

        off_button = QtWidgets.QPushButton("Peltier off")
        off_button.clicked.connect(self.stop_peltier)
        control_layout.addWidget(off_button, 1, 0, 1, 2)

        layout.addWidget(control_box)

        self.status_label = QtWidgets.QLabel("Status: ready. Pins ENA=32, EN1=33, EN2=25.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.local_log = QtWidgets.QPlainTextEdit()
        self.local_log.setReadOnly(True)
        self.local_log.setMinimumHeight(160)
        layout.addWidget(self.local_log, stretch=1)

        self.update_preview()

    def update_preview(self) -> None:
        pwm_value = self.pwm_spin.value()
        battery_equivalent = 7.4 * (pwm_value / 255.0)
        on_time = self.on_time_spin.value()
        off_time = self.off_time_spin.value()
        duty_percent = 100.0 * on_time / max(0.1, on_time + off_time)
        self.preview_label.setText(
            f"PWM {pwm_value}/255 gives approximately {battery_equivalent:.2f} V average "
            f"from 7.4 V. Duty cycle: {on_time:.1f} s on, {off_time:.1f} s off "
            f"({duty_percent:.1f}% active)."
        )

    def _send(self, command: str) -> None:
        self.command_requested.emit(command)
        self.local_log.appendPlainText(f"> {command}")
        scrollbar = self.local_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start_continuous(self) -> None:
        pwm_value = self.pwm_spin.value()
        if pwm_value <= 0:
            self.stop_peltier()
            return

        self.phase_timer.stop()
        self.cycle_running = False
        self.cycle_phase = "ON"
        mode = self.mode_combo.currentData()
        self._send(f"PELTIER {mode} {pwm_value}")
        self.last_direction = mode
        self.status_label.setText(f"Status: continuous {self.mode_combo.currentText().lower()} with PWM {pwm_value}.")

    def start_duty_cycle(self) -> None:
        pwm_value = self.pwm_spin.value()
        if pwm_value <= 0:
            self.stop_peltier()
            return

        self.phase_timer.stop()
        self.cycle_running = False
        self.cycle_mode = self.mode_combo.currentData()
        on_ms = int(round(self.on_time_spin.value() * 1000))
        off_ms = int(round(self.off_time_spin.value() * 1000))
        self._send(f"PELTIER CYCLE {self.cycle_mode} {pwm_value} {on_ms} {off_ms}")
        self.last_direction = self.cycle_mode
        self.status_label.setText(
            f"Status: ESP32 duty cycle active, {self.mode_combo.currentText().lower()} "
            f"{self.on_time_spin.value():.1f} s on / {self.off_time_spin.value():.1f} s off."
        )

    def _start_on_phase(self) -> None:
        if not self.cycle_running:
            return

        pwm_value = self.pwm_spin.value()
        self._send(f"PELTIER {self.cycle_mode} {pwm_value}")
        self.last_direction = self.cycle_mode
        self.cycle_phase = "ON"
        duration_ms = int(round(self.on_time_spin.value() * 1000))
        self.status_label.setText(
            f"Status: ON-phase {self.mode_combo.currentText().lower()} for {self.on_time_spin.value():.1f} s."
        )
        self.phase_timer.start(duration_ms)

    def _start_off_phase(self, reason: str = "Uit-fase.") -> None:
        if not self.cycle_running:
            return

        self._send("PELTIER OFF")
        self.cycle_phase = "OFF"
        duration_ms = int(round(self.off_time_spin.value() * 1000))
        self.status_label.setText(f"Status: {reason} Peltier off for {self.off_time_spin.value():.1f} s.")
        self.phase_timer.start(duration_ms)

    def _advance_cycle_phase(self) -> None:
        if not self.cycle_running:
            return

        if self.cycle_phase == "ON":
            self._start_off_phase()
        else:
            self._start_on_phase()

    def stop_peltier(self) -> None:
        self.phase_timer.stop()
        self.cycle_running = False
        self.cycle_phase = "OFF"
        self._send("PELTIER CYCLE OFF")
        self.status_label.setText("Status: Peltier off.")


class RawSpectrumComparisonWindow(QtWidgets.QWidget):
    command_requested = QtCore.pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Raw spectrum comparison")
        self.resize(1850, 1000)
        self.latest_raw_values = np.zeros(18, dtype=float)
        self.latest_raw_timestamp_ms = 0
        self.composite_values = np.full(18, np.nan, dtype=float)
        self.channel_checkboxes = []
        self.saved_sample_curves = []
        self.saved_sample_counter = 1

        self.setStyleSheet("""
            QWidget {
                background-color: white;
                color: #111827;
            }
            QGroupBox {
                border: 1px solid #111827;
                margin-top: 10px;
                padding: 8px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                background-color: white;
            }
            QPushButton {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #111827;
                padding: 6px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
            }
            QLineEdit, QSpinBox {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #111827;
                padding: 4px;
            }
            QCheckBox {
                color: #111827;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter, stretch=1)

        graph_panel = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout(graph_panel)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(graph_panel)
        splitter.setStretchFactor(0, 5)

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(420)
        controls_panel = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)
        controls_scroll.setWidget(controls_panel)
        splitter.addWidget(controls_scroll)
        splitter.setStretchFactor(1, 1)

        dimensions_box = QtWidgets.QGroupBox("Dimensions")
        dimensions_layout = QtWidgets.QGridLayout(dimensions_box)
        self.plot_height_spin = QtWidgets.QSpinBox()
        self.plot_height_spin.setRange(240, 1200)
        self.plot_height_spin.setValue(640)
        self.plot_height_spin.setSuffix(" px")
        self.plot_height_spin.valueChanged.connect(self.update_dimensions)
        dimensions_layout.addWidget(QtWidgets.QLabel("Plot height"), 0, 0)
        dimensions_layout.addWidget(self.plot_height_spin, 0, 1)
        controls_layout.addWidget(dimensions_box)

        self.export_widget = QtWidgets.QWidget()
        export_layout = QtWidgets.QVBoxLayout(self.export_widget)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(6)

        self.plot = pg.PlotWidget(title="Raw spectrum per LED")
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.0)
        self.plot.getPlotItem().setContentsMargins(18, 18, 18, 18)
        self.plot.getAxis("left").setPen(pg.mkPen("#111827", width=2))
        self.plot.getAxis("bottom").setPen(pg.mkPen("#111827", width=2))
        self.plot.getAxis("left").setTextPen(pg.mkPen("#111827"))
        self.plot.getAxis("bottom").setTextPen(pg.mkPen("#111827"))
        self.plot.getAxis("left").setLabel("Raw value (counts)", color="#111827")
        self.plot.getAxis("bottom").setLabel("Wavelength (nm)", color="#111827")
        self.plot.setXRange(float(WAVELENGTHS[0]), float(WAVELENGTHS[-1]), padding=0.02)
        self.spectral_background = SpectrumChannelBandsItem(WAVELENGTHS)
        self.spectral_background.setZValue(-100)
        self.plot.addItem(self.spectral_background)
        self.grid_item = SpectrumGridItem([410, 485, 560, 645, 730, 810, 900, 940])
        self.grid_item.setZValue(-10)
        self.plot.addItem(self.grid_item)
        self.legend = self.plot.addLegend(offset=(10, 10))
        self.preview_curve = self.plot.plot(
            WAVELENGTHS,
            self.composite_values,
            pen=pg.mkPen("#000000", width=2),
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush("#000000"),
            symbolPen=pg.mkPen("#000000"),
            name="Preview raw sample",
        )
        export_layout.addWidget(self.plot, stretch=1)
        graph_layout.addWidget(self.export_widget, stretch=1)
        self.update_dimensions()

        led_control_box = QtWidgets.QGroupBox("LED source")
        led_control_layout = QtWidgets.QGridLayout(led_control_box)
        for column, (title, command) in enumerate([
            ("UV only", "UV ONLY"),
            ("White only", "WHITE ONLY"),
            ("IR only", "NIR ONLY"),
            ("All off", "ALL OFF"),
        ]):
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(lambda _checked=False, cmd=command: self.command_requested.emit(cmd))
            led_control_layout.addWidget(button, 0, column)
        controls_layout.addWidget(led_control_box)

        channel_box = QtWidgets.QGroupBox("Add channels to raw spectrum")
        channel_layout = QtWidgets.QVBoxLayout(channel_box)
        selection_grid = QtWidgets.QGridLayout()
        for idx, label in enumerate(CHANNEL_LABELS):
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setChecked(idx < 6)
            self.channel_checkboxes.append(checkbox)
            selection_grid.addWidget(checkbox, idx // 3, idx % 3)
        channel_layout.addLayout(selection_grid)

        preset_grid = QtWidgets.QGridLayout()
        for column, (title, indices) in enumerate([
            ("Select UV 6", range(0, 6)),
            ("Select white 6", range(6, 12)),
            ("Select IR 6", range(12, 18)),
            ("Select all", range(18)),
        ]):
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(lambda _checked=False, selected=tuple(indices): self.set_channel_selection(selected))
            preset_grid.addWidget(button, 0, column)
        channel_layout.addLayout(preset_grid)

        save_selected_button = QtWidgets.QPushButton("Save selected raw channels")
        save_selected_button.clicked.connect(self.save_selected_channels)
        channel_layout.addWidget(save_selected_button)
        controls_layout.addWidget(channel_box)

        self.status_label = QtWidgets.QLabel("No raw LED segments saved yet.")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)

        sample_box = QtWidgets.QGroupBox("Compare raw samples")
        sample_layout = QtWidgets.QGridLayout(sample_box)
        self.sample_name_edit = QtWidgets.QLineEdit("Raw sample 1")
        self.sample_name_edit.setPlaceholderText("E.g. Blank raw, Sample A raw...")
        sample_layout.addWidget(QtWidgets.QLabel("Sample name"), 0, 0)
        sample_layout.addWidget(self.sample_name_edit, 0, 1)

        add_sample_button = QtWidgets.QPushButton("Add this raw line to comparison")
        add_sample_button.clicked.connect(self.add_current_sample_curve)
        sample_layout.addWidget(add_sample_button, 1, 0, 1, 2)

        clear_samples_button = QtWidgets.QPushButton("Clear raw sample comparison")
        clear_samples_button.clicked.connect(self.clear_sample_curves)
        sample_layout.addWidget(clear_samples_button, 2, 0, 1, 2)
        controls_layout.addWidget(sample_box)

        export_box = QtWidgets.QGroupBox("PNG export")
        png_layout = QtWidgets.QGridLayout(export_box)
        self.png_name_edit = QtWidgets.QLineEdit("raw_spectrum")
        png_layout.addWidget(QtWidgets.QLabel("File name"), 0, 0)
        png_layout.addWidget(self.png_name_edit, 0, 1)
        png_button = QtWidgets.QPushButton("Save PNG")
        png_button.clicked.connect(self.save_png)
        png_layout.addWidget(png_button, 1, 0, 1, 2)
        controls_layout.addWidget(export_box)

        clear_button = QtWidgets.QPushButton("Clear current raw spectrum")
        clear_button.clicked.connect(self.clear_current_spectrum)
        controls_layout.addWidget(clear_button)
        controls_layout.addStretch(1)

    def update_raw_values(self, frame: SensorFrame) -> None:
        self.latest_raw_values = np.array(frame.values, dtype=float)
        self.latest_raw_timestamp_ms = frame.timestamp_ms

    def set_channel_selection(self, selected_indices) -> None:
        selected_set = set(selected_indices)
        for idx, checkbox in enumerate(self.channel_checkboxes):
            checkbox.setChecked(idx in selected_set)

    def save_selected_channels(self) -> None:
        selected_indices = [idx for idx, checkbox in enumerate(self.channel_checkboxes) if checkbox.isChecked()]
        if not selected_indices:
            QtWidgets.QMessageBox.warning(self, "No channels", "Select at least one channel to save.")
            return

        self.composite_values[selected_indices] = self.latest_raw_values[selected_indices]
        self.preview_curve.setData(WAVELENGTHS, self.composite_values)
        self.update_range()
        labels = ", ".join(str(int(WAVELENGTHS[idx])) for idx in selected_indices)
        self.status_label.setText(
            f"Raw channels saved: {labels} nm at {self.latest_raw_timestamp_ms} ms."
        )

    def update_dimensions(self) -> None:
        self.plot.setMinimumHeight(self.plot_height_spin.value())
        self.plot.setMaximumHeight(self.plot_height_spin.value())
        self.update_range()
        self.export_widget.adjustSize()

    def update_range(self) -> None:
        all_values = [self.composite_values[np.isfinite(self.composite_values)]]
        for _name, _curve, values in self.saved_sample_curves:
            all_values.append(values[np.isfinite(values)])
        finite_values = np.concatenate([values for values in all_values if values.size]) if any(values.size for values in all_values) else np.array([])
        max_value = float(np.max(finite_values)) if finite_values.size else 1.0
        y_max = max(max_value * 1.12, 1.0)
        self.plot.setYRange(0.0, y_max, padding=0)
        self.spectral_background.setYMax(y_max)
        self.grid_item.setYMax(y_max)

    def refresh_legend(self) -> None:
        self.legend.clear()
        for name, curve, _values in self.saved_sample_curves:
            self.legend.addItem(curve, name)

    def add_current_sample_curve(self) -> None:
        values = np.array(self.composite_values, dtype=float)
        if not np.isfinite(values).any():
            QtWidgets.QMessageBox.warning(self, "No raw sample", "Save at least one raw channel for this sample first.")
            return

        name = self.sample_name_edit.text().strip() or f"Raw sample {self.saved_sample_counter}"
        color = pg.intColor(len(self.saved_sample_curves), hues=12, values=1.0, maxValue=220)
        curve = self.plot.plot(
            WAVELENGTHS,
            values,
            pen=pg.mkPen(color, width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(color),
            symbolPen=pg.mkPen(color),
            name=name,
        )
        self.saved_sample_curves.append((name, curve, values))
        self.saved_sample_counter += 1
        self.sample_name_edit.setText(f"Raw sample {self.saved_sample_counter}")
        self.composite_values[:] = np.nan
        self.preview_curve.setData(WAVELENGTHS, self.composite_values)
        self.refresh_legend()
        self.update_range()
        self.status_label.setText(f"Raw sample added to comparison: {name}. Build the next raw sample now.")

    def clear_sample_curves(self) -> None:
        for _name, curve, _values in self.saved_sample_curves:
            self.plot.removeItem(curve)
        self.saved_sample_curves.clear()
        self.saved_sample_counter = 1
        self.sample_name_edit.setText("Raw sample 1")
        self.refresh_legend()
        self.update_range()
        self.status_label.setText("Raw sample comparison cleared.")

    def save_png(self) -> None:
        raw_name = self.png_name_edit.text().strip()
        if not raw_name:
            QtWidgets.QMessageBox.warning(self, "File name", "Enter a file name first.")
            return

        safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in raw_name)
        if not safe_name.lower().endswith(".png"):
            safe_name += ".png"

        output_path = Path(__file__).resolve().parent / safe_name
        pixmap = self.export_widget.grab()
        if pixmap.save(str(output_path), "PNG"):
            self.status_label.setText(f"Raw PNG saved: {output_path}")
        else:
            QtWidgets.QMessageBox.warning(self, "PNG export", "Failed to save PNG.")

    def clear_current_spectrum(self) -> None:
        self.composite_values[:] = np.nan
        self.preview_curve.setData(WAVELENGTHS, self.composite_values)
        self.update_range()
        self.status_label.setText("Current raw spectrum cleared.")


class CalibrationWindow(QtWidgets.QWidget):
    calibration_changed = QtCore.pyqtSignal()

    PROFILES = [
        ("UV LED", "UV"),
        ("White LED", "WHITE"),
        ("IR LED", "IR"),
    ]
    ABSORBANCE_PROFILE_CHANNELS = {
        "UV": tuple(range(0, 6)),
        "WHITE": tuple(range(6, 12)),
        "IR": tuple(range(12, 18)),
    }

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("System calibration 18 channels")
        self.resize(980, 780)
        self.latest_values = np.zeros(18, dtype=float)
        self.latest_timestamp_ms = 0
        self.calibration = {
            profile_key: {
                "dark": None,
                "reference": None,
                "active": False,
                "calibrated": np.full(18, np.nan, dtype=float),
                "calibrated_mode": "",
                "calibrated_timestamp_ms": 0,
                "dark_timestamp_ms": 0,
                "reference_timestamp_ms": 0,
            }
            for _profile_name, profile_key in self.PROFILES
        }
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                color: #111827;
            }
            QGroupBox {
                border: 1px solid #111827;
                margin-top: 10px;
                padding: 8px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                background-color: white;
            }
            QPushButton {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #111827;
                padding: 7px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
            }
            QComboBox, QLineEdit {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #111827;
                padding: 5px;
            }
            QPlainTextEdit {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #111827;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "This page stores dark, blank/reference and calibrated values separately per LED profile. "
            "Each profile always stores all 18 channels as a fixed snapshot, independent of the other profiles."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        steps_box = QtWidgets.QGroupBox("Step by step")
        steps_layout = QtWidgets.QVBoxLayout(steps_box)
        steps_text = QtWidgets.QLabel(
            "1. Under 'Profile to calibrate', choose which LED profile you want to save now.\n"
            "2. Turn all LEDs off, place the same cuvette/geometry and click 'Step 1: save dark'. This stores all 18 channels for this profile only.\n"
            "3. Turn the same LED on, place blank/reference and click 'Step 2: save reference'. This stores all 18 channels again for this profile only.\n"
            "4. Then click 'Use UV', 'Use white LED' or 'Use IR' to apply that saved profile in the graphs.\n"
            "5. Repeat these steps per profile. UV, white LED and IR remain independently stored and do not affect each other."
        )
        steps_text.setWordWrap(True)
        steps_layout.addWidget(steps_text)
        layout.addWidget(steps_box)

        controls_box = QtWidgets.QGroupBox("Calibration controls")
        controls_layout = QtWidgets.QGridLayout(controls_box)

        self.profile_combo = QtWidgets.QComboBox()
        for profile_name, profile_key in self.PROFILES:
            self.profile_combo.addItem(profile_name, profile_key)
        self.profile_combo.currentIndexChanged.connect(self.on_profile_changed)
        controls_layout.addWidget(QtWidgets.QLabel("Profile to calibrate"), 0, 0)
        controls_layout.addWidget(self.profile_combo, 0, 1)

        self.active_profiles_label = QtWidgets.QLabel("Active profile for graphs: none")
        self.active_profiles_label.setWordWrap(True)
        controls_layout.addWidget(self.active_profiles_label, 1, 0, 1, 2)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Normalized (sample-dark)/(blank-dark)", "normalized")
        self.mode_combo.addItem("Absorbance -log10(normalized)", "absorbance")
        self.mode_combo.addItem("Dark corrected sample-dark", "dark_corrected")
        self.mode_combo.addItem("No system calibration", "none")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        controls_layout.addWidget(QtWidgets.QLabel("Output for graphs"), 2, 0)
        controls_layout.addWidget(self.mode_combo, 2, 1)

        self.raw_while_editing_checkbox = QtWidgets.QCheckBox("Show normal values while calibrating another profile")
        self.raw_while_editing_checkbox.setChecked(True)
        self.raw_while_editing_checkbox.toggled.connect(self.on_preview_mode_changed)
        controls_layout.addWidget(self.raw_while_editing_checkbox, 3, 0, 1, 2)

        use_uv_button = QtWidgets.QPushButton("Use UV calibration")
        use_uv_button.clicked.connect(lambda: self.activate_profile("UV"))
        controls_layout.addWidget(use_uv_button, 4, 0)

        use_white_button = QtWidgets.QPushButton("Use white LED calibration")
        use_white_button.clicked.connect(lambda: self.activate_profile("WHITE"))
        controls_layout.addWidget(use_white_button, 4, 1)

        use_ir_button = QtWidgets.QPushButton("Use IR calibration")
        use_ir_button.clicked.connect(lambda: self.activate_profile("IR"))
        controls_layout.addWidget(use_ir_button, 5, 0, 1, 2)

        dark_button = QtWidgets.QPushButton("Step 1: save dark for this profile")
        dark_button.clicked.connect(self.capture_dark)
        controls_layout.addWidget(dark_button, 6, 0)

        reference_button = QtWidgets.QPushButton("Step 2: save reference for this profile")
        reference_button.clicked.connect(self.capture_reference)
        controls_layout.addWidget(reference_button, 6, 1)

        clear_button = QtWidgets.QPushButton("Clear calibration for this profile")
        clear_button.clicked.connect(self.clear_current_profile)
        controls_layout.addWidget(clear_button, 7, 0)

        clear_active_button = QtWidgets.QPushButton("Stop using all calibrations")
        clear_active_button.clicked.connect(self.deactivate_all_profiles)
        controls_layout.addWidget(clear_active_button, 7, 1)

        layout.addWidget(controls_box)

        self.status_label = QtWidgets.QLabel("No calibration saved yet.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.table = QtWidgets.QTableWidget(18, 6)
        self.table.setHorizontalHeaderLabels(["Channel", "Active", "Live measurement", "Dark", "Reference", "Saved calibration"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        for row, label in enumerate(CHANNEL_LABELS):
            item = QtWidgets.QTableWidgetItem(label)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, item)
        layout.addWidget(self.table, stretch=1)
        self.refresh_table()

    def active_profile(self) -> str:
        return self.profile_combo.currentData()

    def profile_display_name(self, profile_key: str) -> str:
        for profile_name, key in self.PROFILES:
            if key == profile_key:
                return profile_name
        return profile_key

    def active_mode(self) -> str:
        return self.mode_combo.currentData()

    def on_mode_changed(self) -> None:
        self.deactivate_incomplete_profiles()
        self.calibration_changed.emit()

    def on_preview_mode_changed(self) -> None:
        self.calibration_changed.emit()

    def current_profile_ready_for_mode(self) -> bool:
        return self.profile_ready_for_mode(self.active_profile())

    def profile_ready_for_mode(self, profile_key: str) -> bool:
        profile_data = self.calibration[profile_key]
        mode = self.active_mode()
        if mode == "none":
            return True
        dark = profile_data["dark"]
        reference = profile_data["reference"]
        if dark is None or not np.isfinite(dark).all():
            return False
        if mode == "dark_corrected":
            return True
        return reference is not None and np.isfinite(reference).all()

    def set_active_profile(self, profile_key: str) -> None:
        for index in range(self.profile_combo.count()):
            if self.profile_combo.itemData(index) == profile_key:
                self.profile_combo.setCurrentIndex(index)
                return

    def on_profile_changed(self) -> None:
        self.refresh_table()
        self.calibration_changed.emit()

    def emit_if_active(self) -> None:
        if self.active_profile_keys():
            self.calibration_changed.emit()

    def active_profile_keys(self) -> list[str]:
        return [profile_key for _profile_name, profile_key in self.PROFILES if self.calibration[profile_key]["active"]]

    def active_graph_profile(self) -> str | None:
        active_profiles = self.active_profile_keys()
        return active_profiles[0] if active_profiles else None

    def update_active_profiles_label(self) -> None:
        names = [self.profile_display_name(profile_key) for profile_key in self.active_profile_keys()]
        self.active_profiles_label.setText(
            "Active profile for graphs: " + (", ".join(names) if names else "none")
        )

    def deactivate_incomplete_profiles(self) -> None:
        changed = False
        for _profile_name, profile_key in self.PROFILES:
            if self.calibration[profile_key]["active"] and not self.profile_ready_for_mode(profile_key):
                self.calibration[profile_key]["active"] = False
                changed = True
        if changed:
            self.update_active_profiles_label()
            self.refresh_table()

    def should_show_raw_while_editing(self) -> bool:
        active_profile = self.active_graph_profile()
        return (
            self.raw_while_editing_checkbox.isChecked()
            and active_profile is not None
            and self.active_profile() != active_profile
        )

    def graph_output_label(self) -> str:
        mode = self.active_mode()
        active_profile = self.active_graph_profile()
        editing_profile = self.active_profile()
        if mode == "none" or active_profile is None:
            return "Main graph: raw sensor values, no system calibration active."
        active_name = self.profile_display_name(active_profile)
        editing_name = self.profile_display_name(editing_profile)
        if self.should_show_raw_while_editing():
            return (
                f"Main graph: raw sensor values while calibrating {editing_name}. "
                f"Saved active profile: {active_name}, temporarily not applied."
            )
        if mode == "absorbance":
            channels = ", ".join(CHANNEL_LABELS[idx] for idx in self.ABSORBANCE_PROFILE_CHANNELS[active_profile])
            return (
                f"Main graph: absorbance with active profile {active_name}. "
                f"Only these 6 profile channels are shown: {channels}."
            )
        return f"Main graph: {self.mode_combo.currentText()} with active profile {active_name}."

    def calculate_profile_values(self, profile_key: str, values: np.ndarray) -> np.ndarray:
        values = np.array(values, dtype=float)
        mode = self.active_mode()
        profile_data = self.calibration[profile_key]
        dark = profile_data["dark"]
        reference = profile_data["reference"]
        result = np.full(18, np.nan, dtype=float)

        if mode == "none":
            return np.array(values, dtype=float)
        if dark is None:
            return result

        dark_ready_mask = np.isfinite(dark)
        corrected_sample = values[dark_ready_mask] - dark[dark_ready_mask]
        if mode == "dark_corrected":
            result[dark_ready_mask] = corrected_sample
            return result
        if reference is None:
            return result

        ready_mask = dark_ready_mask & np.isfinite(reference)
        corrected_sample = values[ready_mask] - dark[ready_mask]
        corrected_reference = reference[ready_mask] - dark[ready_mask]
        safe_reference = np.where(np.abs(corrected_reference) < 1e-9, np.nan, corrected_reference)
        normalized = corrected_sample / safe_reference
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        if mode == "absorbance":
            profile_indices = np.array(self.ABSORBANCE_PROFILE_CHANNELS[profile_key], dtype=int)
            ready_indices = np.flatnonzero(ready_mask)
            profile_ready_mask = np.isin(ready_indices, profile_indices)
            safe_normalized = np.clip(normalized, 1e-9, None)
            result[ready_indices[profile_ready_mask]] = -np.log10(safe_normalized[profile_ready_mask])
        else:
            result[ready_mask] = normalized
        return result

    def activate_profile(self, profile: str) -> None:
        if not self.profile_ready_for_mode(profile):
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration incomplete",
                f"First take the required dark and blank/reference measurement for {self.profile_display_name(profile)}. "
                "For 'Dark corrected' dark is sufficient; for normalized and absorbance both dark and reference are needed.",
            )
            return

        calibrated_values = self.calculate_profile_values(profile, self.latest_values)
        for _profile_name, profile_key in self.PROFILES:
            self.calibration[profile_key]["active"] = False
        self.calibration[profile]["calibrated"] = np.array(calibrated_values, dtype=float)
        self.calibration[profile]["calibrated_mode"] = self.mode_combo.currentText()
        self.calibration[profile]["calibrated_timestamp_ms"] = self.latest_timestamp_ms
        self.calibration[profile]["active"] = True
        self.update_active_profiles_label()
        self.refresh_table()
        self.calibration_changed.emit()
        self.status_label.setText(
            f"{self.profile_display_name(profile)} saved and active for graphs. "
            f"All 18 calibrated channels stored separately at {self.latest_timestamp_ms} ms."
        )

    def deactivate_all_profiles(self) -> None:
        for _profile_name, profile_key in self.PROFILES:
            self.calibration[profile_key]["active"] = False
        self.update_active_profiles_label()
        self.refresh_table()
        self.calibration_changed.emit()
        self.status_label.setText("All calibrations disabled for graphs.")

    def update_latest_values(self, values: np.ndarray, timestamp_ms: int) -> None:
        self.latest_values = np.array(values, dtype=float)
        self.latest_timestamp_ms = timestamp_ms
        self.refresh_table()

    def capture_dark(self) -> None:
        profile = self.active_profile()
        self.calibration[profile]["active"] = False
        self.calibration[profile]["calibrated"] = np.full(18, np.nan, dtype=float)
        self.calibration[profile]["calibrated_mode"] = ""
        self.calibration[profile]["calibrated_timestamp_ms"] = 0
        self.calibration[profile]["dark"] = np.array(self.latest_values, dtype=float)
        self.calibration[profile]["dark_timestamp_ms"] = self.latest_timestamp_ms
        self.status_label.setText(
            f"Dark saved for {self.profile_combo.currentText()} at {self.latest_timestamp_ms} ms "
            "for all 18 channels. "
            "Then click the appropriate 'Use ... calibration' button to activate this profile."
        )
        self.update_active_profiles_label()
        self.refresh_table()
        self.calibration_changed.emit()

    def capture_reference(self) -> None:
        profile = self.active_profile()
        self.calibration[profile]["active"] = False
        self.calibration[profile]["calibrated"] = np.full(18, np.nan, dtype=float)
        self.calibration[profile]["calibrated_mode"] = ""
        self.calibration[profile]["calibrated_timestamp_ms"] = 0
        self.calibration[profile]["reference"] = np.array(self.latest_values, dtype=float)
        self.calibration[profile]["reference_timestamp_ms"] = self.latest_timestamp_ms
        self.status_label.setText(
            f"Blank/reference saved for {self.profile_combo.currentText()} at {self.latest_timestamp_ms} ms "
            "for all 18 channels. "
            "Then click the appropriate 'Use ... calibration' button to activate this profile."
        )
        self.update_active_profiles_label()
        self.refresh_table()
        self.calibration_changed.emit()

    def clear_current_profile(self) -> None:
        profile = self.active_profile()
        self.calibration[profile] = {
            "dark": None,
            "reference": None,
            "active": False,
            "calibrated": np.full(18, np.nan, dtype=float),
            "calibrated_mode": "",
            "calibrated_timestamp_ms": 0,
            "dark_timestamp_ms": 0,
            "reference_timestamp_ms": 0,
        }
        self.status_label.setText(f"Calibration cleared for {self.profile_combo.currentText()}.")
        self.update_active_profiles_label()
        self.refresh_table()
        self.emit_if_active()

    def apply_to_values(self, values: np.ndarray) -> np.ndarray:
        values = np.array(values, dtype=float)
        mode = self.active_mode()
        active_profiles = self.active_profile_keys()
        if mode == "none" or not active_profiles or self.should_show_raw_while_editing():
            return values

        result = np.array(values, dtype=float)
        for profile_key in active_profiles:
            profile_values = self.calculate_profile_values(profile_key, values)
            ready_mask = np.isfinite(profile_values)
            result[ready_mask] = profile_values[ready_mask]
        return result

    def refresh_table(self) -> None:
        profile_data = self.calibration[self.active_profile()]
        dark = profile_data["dark"]
        reference = profile_data["reference"]
        active = profile_data["active"]
        calibrated = profile_data["calibrated"]
        for row in range(18):
            values = [
                "yes" if active else "no",
                f"{self.latest_values[row]:.4f}",
                "--" if dark is None or not np.isfinite(dark[row]) else f"{dark[row]:.4f}",
                "--" if reference is None or not np.isfinite(reference[row]) else f"{reference[row]:.4f}",
                "--" if not np.isfinite(calibrated[row]) else f"{calibrated[row]:.4f}",
            ]
            for column, text in enumerate(values, start=1):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)


class As7265xMonitor(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        if QtWidgets.QApplication.instance() is None:
            raise RuntimeError("QApplication must be created first.")
        super().__init__()
        self.setWindowTitle("AS7265x + TLC59108 + Peltier/Vibrator Monitor")
        self.resize(1500, 920)

        self.reader = SerialReader()
        self.reader.frame_received.connect(self.on_frame_received)
        self.reader.raw_frame_received.connect(self.on_raw_frame_received)
        self.reader.temperature_received.connect(self.on_temperature_received)
        self.reader.message_received.connect(self.append_log)
        self.reader.connection_changed.connect(self.on_connection_changed)
        self.peltier_window = PeltierControlWindow()
        self.peltier_window.command_requested.connect(self.send_command)
        self.raw_spectra_page = RawSpectrumComparisonWindow()
        self.raw_spectra_page.command_requested.connect(self.send_command)
        self.calibration_window = CalibrationWindow()
        self.calibration_window.calibration_changed.connect(self.reprocess_latest_frame)

        self.latest_sensor_values = np.zeros(18, dtype=float)
        self.latest_values = np.zeros(18, dtype=float)
        self.latest_raw_values = np.zeros(18, dtype=float)
        self.latest_timestamp_ms = 0
        self.latest_raw_timestamp_ms = 0
        self.latest_temperature_c = None
        self.latest_temperature_timestamp_ms = 0
        self.max_history = 500
        self.time_history = collections.deque(maxlen=self.max_history)
        self.channel_history = [collections.deque(maxlen=self.max_history) for _ in range(18)]
        self.final_graph_average_history = collections.deque(maxlen=200)
        self.final_graph_sparkfun_history = collections.deque(maxlen=200)
        self.final_graph_raw_history = collections.deque(maxlen=200)
        self.final_graph_average_count = 5
        self.pending_composite_save_indices = []
        self.pending_composite_graph_samples = []
        self.pending_composite_sparkfun_samples = []
        self.pending_composite_raw_samples = []
        self.pending_composite_required_samples = 0
        self.curves = []
        self.channel_checkboxes = []
        self.composite_led_values = np.full(18, np.nan, dtype=float)
        self.composite_sparkfun_values = np.full(18, np.nan, dtype=float)
        self.composite_raw_values = np.full(18, np.nan, dtype=float)
        self.composite_mean_values = np.full(18, np.nan, dtype=float)
        self.composite_std_values = np.full(18, np.nan, dtype=float)
        self.composite_cv_values = np.full(18, np.nan, dtype=float)
        self.composite_dark_values = np.full(18, np.nan, dtype=float)
        self.composite_reference_values = np.full(18, np.nan, dtype=float)
        self.composite_transmission_values = np.full(18, np.nan, dtype=float)
        self.composite_absorbance_values = np.full(18, np.nan, dtype=float)
        self.composite_led_sources = [""] * 18
        self.composite_calibration_profiles = [""] * 18
        self.composite_channel_checkboxes = []
        self.saved_sample_curves = []
        self.saved_sample_exports = []
        self.saved_sample_counter = 1
        self.last_plot_update = 0.0
        self.frame_counter = 0
        self.last_fps_update = time.perf_counter()
        self.stream_fps = 0.0
        self.manual_y_min = 0.0
        self.manual_y_max = 25000.0

        self._build_ui()
        self.refresh_ports()

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self.refresh_stats)
        self.ui_timer.start(150)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        top_controls = QtWidgets.QGroupBox("Connection")
        controls_layout = QtWidgets.QHBoxLayout(top_controls)

        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setMinimumWidth(160)
        controls_layout.addWidget(QtWidgets.QLabel("COM port"))
        controls_layout.addWidget(self.port_combo)

        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_ports)
        controls_layout.addWidget(self.refresh_button)

        self.baud_combo = QtWidgets.QComboBox()
        self.baud_combo.addItems(["115200", "230400", "460800", "921600"])
        self.baud_combo.setCurrentText("921600")
        controls_layout.addWidget(QtWidgets.QLabel("Baudrate"))
        controls_layout.addWidget(self.baud_combo)

        self.interval_spin = QtWidgets.QSpinBox()
        self.interval_spin.setRange(5, 1000)
        self.interval_spin.setValue(25)
        self.interval_spin.setSuffix(" ms")
        controls_layout.addWidget(QtWidgets.QLabel("Stream interval"))
        controls_layout.addWidget(self.interval_spin)

        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_port)
        controls_layout.addWidget(self.connect_button)

        self.disconnect_button = QtWidgets.QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self.disconnect_port)
        controls_layout.addWidget(self.disconnect_button)

        self.start_button = QtWidgets.QPushButton("Start stream")
        self.start_button.clicked.connect(lambda: self.send_command("STREAM ON"))
        controls_layout.addWidget(self.start_button)

        self.stop_button = QtWidgets.QPushButton("Stop stream")
        self.stop_button.clicked.connect(lambda: self.send_command("STREAM OFF"))
        controls_layout.addWidget(self.stop_button)

        self.apply_interval_button = QtWidgets.QPushButton("Apply interval")
        self.apply_interval_button.clicked.connect(self.apply_stream_interval)
        controls_layout.addWidget(self.apply_interval_button)

        self.auto_scale_checkbox = QtWidgets.QCheckBox("Auto scale")
        self.auto_scale_checkbox.setChecked(True)
        self.auto_scale_checkbox.toggled.connect(self.update_plot_scaling)
        controls_layout.addWidget(self.auto_scale_checkbox)

        self.y_min_spin = QtWidgets.QDoubleSpinBox()
        self.y_min_spin.setRange(-1_000_000_000, 1_000_000_000)
        self.y_min_spin.setDecimals(2)
        self.y_min_spin.setSingleStep(10.0)
        self.y_min_spin.setValue(self.manual_y_min)
        self.y_min_spin.valueChanged.connect(self._on_manual_scale_changed)
        controls_layout.addWidget(QtWidgets.QLabel("Y min"))
        controls_layout.addWidget(self.y_min_spin)

        self.y_max_spin = QtWidgets.QDoubleSpinBox()
        self.y_max_spin.setRange(-1_000_000_000, 1_000_000_000)
        self.y_max_spin.setDecimals(2)
        self.y_max_spin.setSingleStep(10.0)
        self.y_max_spin.setValue(self.manual_y_max)
        self.y_max_spin.valueChanged.connect(self._on_manual_scale_changed)
        controls_layout.addWidget(QtWidgets.QLabel("Y max"))
        controls_layout.addWidget(self.y_max_spin)

        self.apply_scale_button = QtWidgets.QPushButton("Apply scale")
        self.apply_scale_button.clicked.connect(self.apply_manual_scale)
        controls_layout.addWidget(self.apply_scale_button)

        controls_layout.addStretch(1)

        self.top_temperature_label = QtWidgets.QLabel("Cuvette: -- deg C")
        self.top_temperature_label.setMinimumWidth(180)
        self.top_temperature_label.setStyleSheet("color: #38bdf8; font-weight: 700;")
        controls_layout.addWidget(self.top_temperature_label)

        self.status_label = QtWidgets.QLabel("Not connected")
        self.status_label.setMinimumWidth(260)
        controls_layout.addWidget(self.status_label)

        root_layout.addWidget(top_controls)

        center_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root_layout.addWidget(center_splitter, stretch=1)

        plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        center_splitter.addWidget(plot_splitter)
        center_splitter.setStretchFactor(0, 4)

        sidebar = QtWidgets.QTabWidget()
        center_splitter.addWidget(sidebar)
        center_splitter.setStretchFactor(1, 1)

        spectrum_panel = QtWidgets.QWidget()
        spectrum_layout = QtWidgets.QVBoxLayout(spectrum_panel)
        spectrum_layout.setContentsMargins(0, 0, 0, 0)
        spectrum_layout.setSpacing(6)

        self.spectrum_plot = pg.PlotWidget(title="Current spectrum")
        self.spectrum_plot.setBackground("#0f172a")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.25)
        self.spectrum_plot.setLabel("left", "System calibrated value")
        self.spectrum_plot.setLabel("bottom", "Wavelength (nm)")
        self.spectrum_plot.setMouseEnabled(x=False, y=True)
        self.spectrum_curve = self.spectrum_plot.plot(
            WAVELENGTHS,
            self.latest_values,
            pen=pg.mkPen("#38bdf8", width=3),
            symbol="o",
            symbolSize=8,
            symbolBrush=pg.mkBrush("#f8fafc"),
        )
        spectrum_layout.addWidget(self.spectrum_plot, stretch=1)

        self.calibration_graph_label = QtWidgets.QLabel(self.calibration_window.graph_output_label())
        self.calibration_graph_label.setWordWrap(True)
        self.calibration_graph_label.setStyleSheet(
            "background-color: #f8fafc; color: #111827; border: 1px solid #111827; padding: 6px; font-weight: 700;"
        )
        spectrum_layout.addWidget(self.calibration_graph_label)

        self.color_strip = SpectralColorStrip()
        spectrum_layout.addWidget(self.color_strip)
        plot_splitter.addWidget(spectrum_panel)

        self.trend_plot = pg.PlotWidget(title="Live trend of all 18 channels")
        self.trend_plot.setBackground("#0f172a")
        self.trend_plot.showGrid(x=True, y=True, alpha=0.25)
        self.trend_plot.setLabel("left", "System calibrated value")
        self.trend_plot.setLabel("bottom", "Time (s)")
        self.trend_plot.addLegend(offset=(10, 10))
        plot_splitter.addWidget(self.trend_plot)

        self.update_plot_scaling()

        for index, label in enumerate(CHANNEL_LABELS):
            curve = self.trend_plot.plot(
                [],
                [],
                pen=pg.mkPen(CHANNEL_COLORS[index], width=2),
                name=label,
            )
            self.curves.append(curve)

        led_tab = QtWidgets.QWidget()
        led_layout = QtWidgets.QVBoxLayout(led_tab)
        led_layout.setContentsMargins(12, 12, 12, 12)
        led_layout.setSpacing(10)

        sensor_box = QtWidgets.QGroupBox("Sensor speed")
        sensor_box_layout = QtWidgets.QGridLayout(sensor_box)

        self.gain_combo = QtWidgets.QComboBox()
        self.gain_combo.addItems(["0 - 1x", "1 - 3.7x", "2 - 16x", "3 - 64x"])
        self.gain_combo.setCurrentIndex(2)
        sensor_box_layout.addWidget(QtWidgets.QLabel("Gain"), 0, 0)
        sensor_box_layout.addWidget(self.gain_combo, 0, 1)

        self.integration_spin = QtWidgets.QSpinBox()
        self.integration_spin.setRange(0, 255)
        self.integration_spin.setValue(4)
        self.integration_spin.setToolTip("2.8ms * (cycles + 1)")
        sensor_box_layout.addWidget(QtWidgets.QLabel("Integration cycles"), 1, 0)
        sensor_box_layout.addWidget(self.integration_spin, 1, 1)

        self.apply_sensor_button = QtWidgets.QPushButton("Apply sensor settings")
        self.apply_sensor_button.clicked.connect(self.apply_sensor_settings)
        sensor_box_layout.addWidget(self.apply_sensor_button, 2, 0, 1, 2)
        led_layout.addWidget(sensor_box)

        led_title = QtWidgets.QLabel("LED control")
        led_title.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Weight.Bold))
        led_layout.addWidget(led_title)

        for label, command_base in [("NIR", "NIR"), ("White", "WHITE"), ("UV", "UV")]:
            row = LedButtonRow(label, command_base)
            row.command_requested.connect(self.send_command)
            led_layout.addWidget(row)

        all_off_button = QtWidgets.QPushButton("All off")
        all_off_button.clicked.connect(lambda: self.send_command("ALL OFF"))
        led_layout.addWidget(all_off_button)

        open_led_spectra_button = QtWidgets.QPushButton("Open LED spectra page")
        open_led_spectra_button.clicked.connect(self.open_led_spectra_page)
        led_layout.addWidget(open_led_spectra_button)

        open_raw_spectra_button = QtWidgets.QPushButton("Open raw values spectra page")
        open_raw_spectra_button.clicked.connect(self.open_raw_spectra_page)
        led_layout.addWidget(open_raw_spectra_button)

        open_calibration_button = QtWidgets.QPushButton("Open calibration window")
        open_calibration_button.clicked.connect(self.open_calibration_page)
        led_layout.addWidget(open_calibration_button)

        open_peltier_button = QtWidgets.QPushButton("Open standalone Peltier window")
        open_peltier_button.clicked.connect(self.open_peltier_page)
        led_layout.addWidget(open_peltier_button)

        soft_reset_button = QtWidgets.QPushButton("Soft Reset")
        soft_reset_button.clicked.connect(lambda: self.send_command("RESET"))
        led_layout.addWidget(soft_reset_button)

        hard_reset_button = QtWidgets.QPushButton("Hard Reset (RST)")
        hard_reset_button.clicked.connect(self.hard_reset)
        hard_reset_button.setStyleSheet("background-color: #dc2626; color: white; font-weight: 600;")
        led_layout.addWidget(hard_reset_button)

        vibrator_box = QtWidgets.QGroupBox("Mini vibration motor via H-bridge")
        vibrator_layout = QtWidgets.QGridLayout(vibrator_box)

        self.vibration_pwm_spin = QtWidgets.QSpinBox()
        self.vibration_pwm_spin.setRange(0, 255)
        self.vibration_pwm_spin.setValue(153)
        self.vibration_pwm_spin.setToolTip("Manual PWM duty cycle for the vibrator on ENB (pin 14).")
        self.vibration_pwm_spin.valueChanged.connect(self.update_vibration_preview)
        vibrator_layout.addWidget(QtWidgets.QLabel("PWM"), 0, 0)
        vibrator_layout.addWidget(self.vibration_pwm_spin, 0, 1)

        self.vibration_pwm_label = QtWidgets.QLabel()
        self.vibration_pwm_label.setWordWrap(True)
        vibrator_layout.addWidget(self.vibration_pwm_label, 1, 0, 1, 2)

        self.vibration_apply_button = QtWidgets.QPushButton("Start/update vibrator")
        self.vibration_apply_button.clicked.connect(self.apply_vibration_control)
        vibrator_layout.addWidget(self.vibration_apply_button, 2, 0, 1, 2)

        self.vibration_off_button = QtWidgets.QPushButton("Vibrator off")
        self.vibration_off_button.clicked.connect(self.stop_vibration)
        vibrator_layout.addWidget(self.vibration_off_button, 3, 0, 1, 2)

        self.vibration_hint_label = QtWidgets.QLabel("Pins: ENB=14, EN3=26, EN4=27. Manual PWM 0-255; with a 5V H-bridge PWM 153 is approximately 3.0V average.")
        self.vibration_hint_label.setWordWrap(True)
        vibrator_layout.addWidget(self.vibration_hint_label, 4, 0, 1, 2)
        led_layout.addWidget(vibrator_box)

        led_layout.addStretch(1)
        sidebar.addTab(led_tab, "LEDs")
        self.update_vibration_preview()

        channels_tab = QtWidgets.QWidget()
        channels_layout = QtWidgets.QVBoxLayout(channels_tab)
        channels_layout.setContentsMargins(12, 12, 12, 12)
        channels_layout.setSpacing(8)

        checkbox_grid = QtWidgets.QGridLayout()
        for idx, label in enumerate(CHANNEL_LABELS):
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.refresh_trend_visibility)
            self.channel_checkboxes.append(checkbox)
            checkbox_grid.addWidget(checkbox, idx // 2, idx % 2)
        channels_layout.addLayout(checkbox_grid)

        clear_button = QtWidgets.QPushButton("Clear graph history")
        clear_button.clicked.connect(self.clear_history)
        channels_layout.addWidget(clear_button)
        channels_layout.addStretch(1)
        sidebar.addTab(channels_tab, "Channels")

        led_spectra_tab = QtWidgets.QWidget()
        led_spectra_tab.setStyleSheet("""
            QWidget {
                background-color: white;
                color: #111827;
            }
            QGroupBox {
                border: 1px solid #111827;
                margin-top: 10px;
                padding: 8px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                background-color: white;
            }
            QPushButton {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #111827;
                padding: 6px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
            }
            QPushButton:pressed {
                background-color: #e5e7eb;
            }
            QLineEdit, QSpinBox {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #111827;
                padding: 4px;
            }
            QCheckBox {
                color: #111827;
            }
        """)
        led_spectra_layout = QtWidgets.QVBoxLayout(led_spectra_tab)
        led_spectra_layout.setContentsMargins(12, 12, 12, 12)
        led_spectra_layout.setSpacing(10)

        led_spectra_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        led_spectra_layout.addWidget(led_spectra_splitter, stretch=1)

        graph_panel = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout(graph_panel)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(8)
        led_spectra_splitter.addWidget(graph_panel)
        led_spectra_splitter.setStretchFactor(0, 5)

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(420)
        controls_panel = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)
        controls_scroll.setWidget(controls_panel)
        led_spectra_splitter.addWidget(controls_scroll)
        led_spectra_splitter.setStretchFactor(1, 1)

        dimensions_box = QtWidgets.QGroupBox("Dimensions")
        dimensions_layout = QtWidgets.QGridLayout(dimensions_box)

        self.composite_plot_height_spin = QtWidgets.QSpinBox()
        self.composite_plot_height_spin.setRange(240, 1200)
        self.composite_plot_height_spin.setValue(640)
        self.composite_plot_height_spin.setSuffix(" px")
        self.composite_plot_height_spin.valueChanged.connect(self.update_composite_dimensions)
        dimensions_layout.addWidget(QtWidgets.QLabel("Plot height"), 0, 0)
        dimensions_layout.addWidget(self.composite_plot_height_spin, 0, 1)

        controls_layout.addWidget(dimensions_box)

        self.composite_export_widget = QtWidgets.QWidget()
        composite_export_layout = QtWidgets.QVBoxLayout(self.composite_export_widget)
        composite_export_layout.setContentsMargins(0, 0, 0, 0)
        composite_export_layout.setSpacing(6)

        self.composite_spectrum_plot = pg.PlotWidget(title="Composite spectrum per LED")
        self.composite_spectrum_plot.setBackground("w")
        self.composite_spectrum_plot.showGrid(x=True, y=True, alpha=0.0)
        self.composite_spectrum_plot.getPlotItem().setContentsMargins(18, 18, 18, 18)
        self.composite_spectrum_plot.getAxis("left").setPen(pg.mkPen("#111827", width=2))
        self.composite_spectrum_plot.getAxis("bottom").setPen(pg.mkPen("#111827", width=2))
        self.composite_spectrum_plot.getAxis("left").setTextPen(pg.mkPen("#111827"))
        self.composite_spectrum_plot.getAxis("bottom").setTextPen(pg.mkPen("#111827"))
        self.composite_spectrum_plot.getAxis("left").setLabel("System calibrated value", color="#111827")
        self.composite_spectrum_plot.getAxis("bottom").setLabel("Wavelength (nm)", color="#111827")
        self.composite_spectrum_plot.setXRange(float(WAVELENGTHS[0]), float(WAVELENGTHS[-1]), padding=0.02)
        self.composite_spectral_background = SpectrumChannelBandsItem(WAVELENGTHS)
        self.composite_spectral_background.setZValue(-100)
        self.composite_spectrum_plot.addItem(self.composite_spectral_background)
        self.composite_grid_item = SpectrumGridItem([410, 485, 560, 645, 730, 810, 900, 940])
        self.composite_grid_item.setZValue(-10)
        self.composite_spectrum_plot.addItem(self.composite_grid_item)
        self.composite_legend = self.composite_spectrum_plot.addLegend(offset=(10, 10))
        self.composite_spectrum_curve = self.composite_spectrum_plot.plot(
            WAVELENGTHS,
            self.composite_led_values,
            pen=pg.mkPen("#000000", width=2),
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush("#000000"),
            symbolPen=pg.mkPen("#000000"),
            name="Preview current sample",
        )
        composite_export_layout.addWidget(self.composite_spectrum_plot, stretch=1)
        graph_layout.addWidget(self.composite_export_widget, stretch=1)
        self.update_composite_dimensions()

        led_control_box = QtWidgets.QGroupBox("LED source")
        led_control_layout = QtWidgets.QGridLayout(led_control_box)
        for column, (title, command) in enumerate([
            ("UV only", "UV ONLY"),
            ("White only", "WHITE ONLY"),
            ("IR only", "NIR ONLY"),
            ("All off", "ALL OFF"),
        ]):
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(lambda _checked=False, cmd=command: self.send_command(cmd))
            led_control_layout.addWidget(button, 0, column)
        controls_layout.addWidget(led_control_box)

        channel_box = QtWidgets.QGroupBox("Add channels to composite spectrum")
        channel_box_layout = QtWidgets.QVBoxLayout(channel_box)

        selection_grid = QtWidgets.QGridLayout()
        for idx, label in enumerate(CHANNEL_LABELS):
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setChecked(idx < 6)
            self.composite_channel_checkboxes.append(checkbox)
            selection_grid.addWidget(checkbox, idx // 3, idx % 3)
        channel_box_layout.addLayout(selection_grid)

        preset_grid = QtWidgets.QGridLayout()
        for column, (title, indices) in enumerate([
            ("Select UV 6", range(0, 6)),
            ("Select white 6", range(6, 12)),
            ("Select IR 6", range(12, 18)),
            ("Select all", range(18)),
        ]):
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(lambda _checked=False, selected=tuple(indices): self.set_composite_channel_selection(selected))
            preset_grid.addWidget(button, 0, column)
        channel_box_layout.addLayout(preset_grid)

        average_grid = QtWidgets.QGridLayout()
        self.final_graph_average_count_spin = QtWidgets.QSpinBox()
        self.final_graph_average_count_spin.setRange(1, 200)
        self.final_graph_average_count_spin.setValue(self.final_graph_average_count)
        self.final_graph_average_count_spin.setSuffix(" values")
        self.final_graph_average_count_spin.valueChanged.connect(lambda _value: self.update_final_graph_average_status())
        average_grid.addWidget(QtWidgets.QLabel("Average when saving"), 0, 0)
        average_grid.addWidget(self.final_graph_average_count_spin, 0, 1)
        channel_box_layout.addLayout(average_grid)

        self.final_graph_average_status_label = QtWidgets.QLabel("Graph save: waiting for Save, new final values will be collected.")
        self.final_graph_average_status_label.setWordWrap(True)
        channel_box_layout.addWidget(self.final_graph_average_status_label)

        self.save_selected_button = QtWidgets.QPushButton("Save selected channels")
        self.save_selected_button.clicked.connect(self.save_selected_composite_channels)
        channel_box_layout.addWidget(self.save_selected_button)
        controls_layout.addWidget(channel_box)

        self.composite_status_label = QtWidgets.QLabel("No LED segments saved yet.")
        self.composite_status_label.setWordWrap(True)
        controls_layout.addWidget(self.composite_status_label)

        sample_box = QtWidgets.QGroupBox("Compare samples")
        sample_layout = QtWidgets.QGridLayout(sample_box)

        self.sample_name_edit = QtWidgets.QLineEdit("Sample 1")
        self.sample_name_edit.setPlaceholderText("E.g. Blank, Sample A, heated 5s...")
        sample_layout.addWidget(QtWidgets.QLabel("Sample name"), 0, 0)
        sample_layout.addWidget(self.sample_name_edit, 0, 1)

        self.sample_color_combo = QtWidgets.QComboBox()
        for color_name, color_hex in [
            ("Black", "#000000"),
            ("Red", "#e11d48"),
            ("Blue", "#2563eb"),
            ("Green", "#16a34a"),
            ("Orange", "#f97316"),
            ("Purple", "#7c3aed"),
            ("Cyan", "#0891b2"),
            ("Magenta", "#db2777"),
            ("Yellow", "#ca8a04"),
            ("Grey", "#4b5563"),
        ]:
            self.sample_color_combo.addItem(color_name, color_hex)
        sample_layout.addWidget(QtWidgets.QLabel("Line color"), 1, 0)
        sample_layout.addWidget(self.sample_color_combo, 1, 1)

        add_sample_button = QtWidgets.QPushButton("Add this line to comparison")
        add_sample_button.clicked.connect(self.add_current_sample_curve)
        sample_layout.addWidget(add_sample_button, 2, 0, 1, 2)

        self.sample_remove_combo = QtWidgets.QComboBox()
        self.sample_remove_combo.addItem("No lines saved")
        sample_layout.addWidget(QtWidgets.QLabel("Remove line"), 3, 0)
        sample_layout.addWidget(self.sample_remove_combo, 3, 1)

        remove_sample_button = QtWidgets.QPushButton("Remove selected line")
        remove_sample_button.clicked.connect(self.remove_selected_sample_curve)
        sample_layout.addWidget(remove_sample_button, 4, 0, 1, 2)

        clear_samples_button = QtWidgets.QPushButton("Clear sample comparison")
        clear_samples_button.clicked.connect(self.clear_sample_curves)
        sample_layout.addWidget(clear_samples_button, 5, 0, 1, 2)
        controls_layout.addWidget(sample_box)

        export_box = QtWidgets.QGroupBox("Export")
        export_layout = QtWidgets.QGridLayout(export_box)
        self.composite_png_name_edit = QtWidgets.QLineEdit("composite_spectrum")
        export_layout.addWidget(QtWidgets.QLabel("File name"), 0, 0)
        export_layout.addWidget(self.composite_png_name_edit, 0, 1)

        export_button = QtWidgets.QPushButton("Save PNG")
        export_button.clicked.connect(self.save_composite_spectrum_png)
        export_layout.addWidget(export_button, 1, 0, 1, 2)

        csv_button = QtWidgets.QPushButton("Save CSV data")
        csv_button.clicked.connect(self.save_composite_spectrum_csv)
        export_layout.addWidget(csv_button, 2, 0, 1, 2)
        controls_layout.addWidget(export_box)

        clear_composite_button = QtWidgets.QPushButton("Clear composite spectrum")
        clear_composite_button.clicked.connect(self.clear_composite_spectrum)
        controls_layout.addWidget(clear_composite_button)
        controls_layout.addStretch(1)
        self.led_spectra_page = led_spectra_tab
        self.led_spectra_page.setWindowTitle("LED spectra export")
        self.led_spectra_page.resize(1850, 1000)

        info_tab = QtWidgets.QWidget()
        info_layout = QtWidgets.QVBoxLayout(info_tab)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setSpacing(10)

        self.stats_label = QtWidgets.QLabel("Samples: 0\nFPS: 0.0\nLast time: 0 ms")
        self.stats_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        info_layout.addWidget(self.stats_label)

        temperature_box = QtWidgets.QGroupBox("Cuvette temperature MCP9808")
        temperature_layout = QtWidgets.QVBoxLayout(temperature_box)

        self.temperature_label = QtWidgets.QLabel("No temperature measurement received yet")
        temperature_font = QtGui.QFont("Segoe UI", 16, QtGui.QFont.Weight.Bold)
        self.temperature_label.setFont(temperature_font)
        self.temperature_label.setStyleSheet("color: #38bdf8;")
        temperature_layout.addWidget(self.temperature_label)

        self.temperature_hint_label = QtWidgets.QLabel("Firmware expects the Adafruit MCP9808 library at I2C address 0x18.")
        self.temperature_hint_label.setWordWrap(True)
        temperature_layout.addWidget(self.temperature_hint_label)

        self.temperature_request_button = QtWidgets.QPushButton("Read temperature now")
        self.temperature_request_button.clicked.connect(lambda: self.send_command("TEMP"))
        temperature_layout.addWidget(self.temperature_request_button)

        info_layout.addWidget(temperature_box)

        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        info_layout.addWidget(self.log_text, stretch=1)
        sidebar.addTab(info_tab, "Status")

    def open_led_spectra_page(self) -> None:
        self.led_spectra_page.show()
        self.led_spectra_page.raise_()
        self.led_spectra_page.activateWindow()

    def open_raw_spectra_page(self) -> None:
        self.raw_spectra_page.show()
        self.raw_spectra_page.raise_()
        self.raw_spectra_page.activateWindow()

    def open_calibration_page(self) -> None:
        self.calibration_window.show()
        self.calibration_window.raise_()
        self.calibration_window.activateWindow()

    def open_peltier_page(self) -> None:
        self.peltier_window.show()
        self.peltier_window.raise_()
        self.peltier_window.activateWindow()

    def refresh_ports(self) -> None:
        current = self.port_combo.currentText()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo.clear()
        self.port_combo.addItems(ports)

        if current and current in ports:
            self.port_combo.setCurrentText(current)

    def connect_port(self) -> None:
        port_name = self.port_combo.currentText().strip()
        if not port_name:
            QtWidgets.QMessageBox.warning(self, "COM port", "Select a COM port first.")
            return

        try:
            baudrate = int(self.baud_combo.currentText())
            self.reader.connect_port(port_name, baudrate)
            time.sleep(0.15)
            self.send_command("HEADER")
            self.apply_stream_interval()
            self.apply_sensor_settings()
            self.stop_peltier()
            self.stop_vibration()
            self.send_command("STREAM ON")
        except (ValueError, serial.SerialException) as exc:
            QtWidgets.QMessageBox.critical(self, "Connection failed", str(exc))

    def disconnect_port(self) -> None:
        self.reader.disconnect_port()

    def on_connection_changed(self, connected: bool, status: str) -> None:
        self.status_label.setText(status)
        self.append_log(status)
        if not connected:
          return

    def send_command(self, command: str) -> None:
        try:
            self.reader.send_command(command)
            self.update_active_calibration_profile_from_command(command)
            self.append_log(f"> {command}")
        except serial.SerialException as exc:
            QtWidgets.QMessageBox.warning(self, "Not connected", str(exc))

    def update_active_calibration_profile_from_command(self, command: str) -> None:
        command = command.strip().upper()
        if command.startswith("UV "):
            return
        elif command.startswith("WHITE "):
            return
        elif command.startswith("NIR "):
            return

    def hard_reset(self) -> None:
        try:
            self.reader.hard_reset()
            self.append_log("> HARD RESET")
        except serial.SerialException as exc:
            QtWidgets.QMessageBox.warning(self, "Not connected", str(exc))

    def apply_stream_interval(self) -> None:
        self.send_command(f"INTERVAL {self.interval_spin.value()}")

    def apply_sensor_settings(self) -> None:
        gain_value = self.gain_combo.currentIndex()
        integration_value = self.integration_spin.value()
        self.send_command(f"GAIN {gain_value}")
        self.send_command(f"INTEGRATION {integration_value}")

    def update_peltier_preview(self) -> None:
        self.peltier_window.update_preview()

    def apply_peltier_control(self) -> None:
        self.peltier_window.start_continuous()

    def apply_peltier_cycle(self) -> None:
        self.peltier_window.start_duty_cycle()

    def stop_peltier(self) -> None:
        self.peltier_window.stop_peltier()

    def update_vibration_preview(self) -> None:
        pwm_value = self.vibration_pwm_spin.value()
        effective_voltage = VIBRATOR_SUPPLY_VOLTAGE * (pwm_value / 255.0)
        self.vibration_pwm_label.setText(
            f"PWM to ENB: {pwm_value}/255, approximately {effective_voltage:.2f} V average "
            f"on a 5V H-bridge."
        )

    def apply_vibration_control(self) -> None:
        pwm_value = self.vibration_pwm_spin.value()
        if pwm_value <= 0:
            self.stop_vibration()
            return

        self.send_command(f"VIBRATOR {pwm_value}")

    def stop_vibration(self) -> None:
        self.send_command("VIBRATOR OFF")

    def _on_manual_scale_changed(self) -> None:
        self.manual_y_min = self.y_min_spin.value()
        self.manual_y_max = self.y_max_spin.value()
        if not self.auto_scale_checkbox.isChecked():
            self.update_plot_scaling()

    def apply_manual_scale(self) -> None:
        self.manual_y_min = self.y_min_spin.value()
        self.manual_y_max = self.y_max_spin.value()

        if self.manual_y_min >= self.manual_y_max:
            QtWidgets.QMessageBox.warning(self, "Invalid scale", "Y min must be less than Y max.")
            return

        self.auto_scale_checkbox.setChecked(False)
        self.update_plot_scaling()

    def remember_final_graph_frame(self) -> None:
        self.final_graph_average_history.append(np.array(self.latest_values, dtype=float))
        self.final_graph_sparkfun_history.append(np.array(self.latest_sensor_values, dtype=float))
        self.final_graph_raw_history.append(np.array(self.latest_raw_values, dtype=float))
        self.update_final_graph_average_status()

    def average_recent_frames(self, history: collections.deque) -> np.ndarray:
        count = self.final_graph_average_count_spin.value()
        recent_values = list(history)[-count:]
        if not recent_values:
            return np.full(18, np.nan, dtype=float)
        return np.mean(np.vstack(recent_values), axis=0)

    def update_final_graph_average_status(self) -> None:
        if not hasattr(self, "final_graph_average_status_label"):
            return
        count = self.final_graph_average_count_spin.value()
        if self.pending_composite_required_samples:
            available = len(self.pending_composite_graph_samples)
            self.final_graph_average_status_label.setText(
                f"Graph save in progress: {available}/{self.pending_composite_required_samples} new final values collected."
            )
            return
        self.final_graph_average_status_label.setText(
            f"Graph save: waiting for Save, then {count} new final values will be averaged."
        )

    def collect_pending_composite_save_frame(self) -> None:
        if not self.pending_composite_required_samples:
            return

        self.pending_composite_graph_samples.append(np.array(self.latest_values, dtype=float))
        self.pending_composite_sparkfun_samples.append(np.array(self.latest_sensor_values, dtype=float))
        self.pending_composite_raw_samples.append(np.array(self.latest_raw_values, dtype=float))
        collected = len(self.pending_composite_graph_samples)
        self.update_final_graph_average_status()

        if collected < self.pending_composite_required_samples:
            return

        selected_indices = list(self.pending_composite_save_indices)
        required_samples = self.pending_composite_required_samples
        graph_sample_matrix = np.vstack(self.pending_composite_graph_samples)
        averaged_graph_values = np.mean(graph_sample_matrix, axis=0)
        std_graph_values = np.std(graph_sample_matrix, axis=0, ddof=1) if required_samples > 1 else np.zeros(18, dtype=float)
        cv_graph_values = np.full(18, np.nan, dtype=float)
        valid_mean = np.isfinite(averaged_graph_values) & (np.abs(averaged_graph_values) >= 1e-12)
        cv_graph_values[valid_mean] = (std_graph_values[valid_mean] / np.abs(averaged_graph_values[valid_mean])) * 100.0
        averaged_sparkfun_values = np.mean(np.vstack(self.pending_composite_sparkfun_samples), axis=0)
        averaged_raw_values = np.mean(np.vstack(self.pending_composite_raw_samples), axis=0)
        self.pending_composite_save_indices = []
        self.pending_composite_graph_samples = []
        self.pending_composite_sparkfun_samples = []
        self.pending_composite_raw_samples = []
        self.pending_composite_required_samples = 0
        self.finalize_composite_channel_save(
            selected_indices,
            averaged_graph_values,
            averaged_sparkfun_values,
            averaged_raw_values,
            std_graph_values,
            cv_graph_values,
            required_samples,
        )
        self.update_final_graph_average_status()

    def update_plot_scaling(self) -> None:
        auto_scale = self.auto_scale_checkbox.isChecked()

        self.y_min_spin.setEnabled(not auto_scale)
        self.y_max_spin.setEnabled(not auto_scale)
        self.apply_scale_button.setEnabled(not auto_scale)

        if auto_scale:
            self.spectrum_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            self.trend_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            return

        if self.manual_y_min >= self.manual_y_max:
            return

        self.spectrum_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        self.trend_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        self.spectrum_plot.setYRange(self.manual_y_min, self.manual_y_max, padding=0)
        self.trend_plot.setYRange(self.manual_y_min, self.manual_y_max, padding=0)

    def on_frame_received(self, frame: SensorFrame) -> None:
        self.latest_sensor_values = np.array(frame.values, dtype=float)
        self.calibration_window.update_latest_values(self.latest_sensor_values, frame.timestamp_ms)
        self.latest_values = self.calibration_window.apply_to_values(self.latest_sensor_values)
        self.latest_timestamp_ms = frame.timestamp_ms
        self.remember_final_graph_frame()
        self.collect_pending_composite_save_frame()

        time_seconds = frame.timestamp_ms / 1000.0
        self.time_history.append(time_seconds)
        for idx, value in enumerate(self.latest_values):
            self.channel_history[idx].append(value)

        self.frame_counter += 1
        now = time.perf_counter()
        elapsed = now - self.last_fps_update
        if elapsed >= 0.5:
            self.stream_fps = self.frame_counter / elapsed
            self.frame_counter = 0
            self.last_fps_update = now

        self.refresh_plots()

    def reprocess_latest_frame(self) -> None:
        self.final_graph_average_history.clear()
        self.final_graph_sparkfun_history.clear()
        self.final_graph_raw_history.clear()
        self.latest_values = self.calibration_window.apply_to_values(self.latest_sensor_values)
        self.remember_final_graph_frame()
        self.calibration_graph_label.setText(self.calibration_window.graph_output_label())
        self.spectrum_curve.setData(WAVELENGTHS, self.latest_values)
        self.color_strip.update_values(self.latest_values)
        if not self.auto_scale_checkbox.isChecked():
            self.update_plot_scaling()

    def on_raw_frame_received(self, frame: SensorFrame) -> None:
        self.latest_raw_values = frame.values
        self.latest_raw_timestamp_ms = frame.timestamp_ms
        self.raw_spectra_page.update_raw_values(frame)

    def on_temperature_received(self, frame: TemperatureFrame) -> None:
        self.latest_temperature_c = frame.temperature_c
        self.latest_temperature_timestamp_ms = frame.timestamp_ms
        temperature_text = f"{frame.temperature_c:.3f} deg C"
        self.temperature_label.setText(temperature_text)
        self.top_temperature_label.setText(f"Cuvette: {temperature_text}")

    def refresh_plots(self) -> None:
        now = time.perf_counter()
        if now - self.last_plot_update < 0.02:
            return

        self.last_plot_update = now
        self.calibration_graph_label.setText(self.calibration_window.graph_output_label())
        self.spectrum_curve.setData(WAVELENGTHS, self.latest_values)
        self.color_strip.update_values(self.latest_values)
        if not self.auto_scale_checkbox.isChecked():
            self.update_plot_scaling()

        if not self.time_history:
            return

        times = np.array(self.time_history, dtype=float)
        times = times - times[-1]

        for idx, curve in enumerate(self.curves):
            if self.channel_checkboxes[idx].isChecked():
                values = np.array(self.channel_history[idx], dtype=float)
                curve.setData(times, values)
                curve.show()
            else:
                curve.hide()

    def refresh_trend_visibility(self) -> None:
        self.refresh_plots()

    def set_composite_channel_selection(self, selected_indices) -> None:
        selected_set = set(selected_indices)
        for idx, checkbox in enumerate(self.composite_channel_checkboxes):
            checkbox.setChecked(idx in selected_set)

    def save_selected_composite_channels(self) -> None:
        if self.pending_composite_required_samples:
            QtWidgets.QMessageBox.warning(
                self,
                "Averaging in progress",
                "An averaging save is already running. Wait until all values are collected.",
            )
            return

        selected_indices = [
            idx for idx, checkbox in enumerate(self.composite_channel_checkboxes)
            if checkbox.isChecked()
        ]
        if not selected_indices:
            QtWidgets.QMessageBox.warning(self, "No channels", "Select at least one channel to save.")
            return

        self.pending_composite_save_indices = selected_indices
        self.pending_composite_graph_samples = []
        self.pending_composite_sparkfun_samples = []
        self.pending_composite_raw_samples = []
        self.pending_composite_required_samples = self.final_graph_average_count_spin.value()
        labels = ", ".join(str(int(WAVELENGTHS[idx])) for idx in selected_indices)
        self.composite_status_label.setText(
            f"Averaging started for channels {labels} nm. "
            f"Now collecting {self.pending_composite_required_samples} new final graph values..."
        )
        self.update_final_graph_average_status()

    def finalize_composite_channel_save(
        self,
        selected_indices,
        averaged_graph_values: np.ndarray,
        averaged_sparkfun_values: np.ndarray,
        averaged_raw_values: np.ndarray,
        std_graph_values: np.ndarray,
        cv_graph_values: np.ndarray,
        sample_count: int,
    ) -> None:
        self.composite_led_values[selected_indices] = averaged_graph_values[selected_indices]
        self.composite_sparkfun_values[selected_indices] = averaged_sparkfun_values[selected_indices]
        self.composite_raw_values[selected_indices] = averaged_raw_values[selected_indices]
        self.composite_mean_values[selected_indices] = averaged_graph_values[selected_indices]
        self.composite_std_values[selected_indices] = std_graph_values[selected_indices]
        self.composite_cv_values[selected_indices] = cv_graph_values[selected_indices]
        self.store_composite_calibration_columns(selected_indices, averaged_sparkfun_values)
        source_label = self._selected_composite_source_label(selected_indices)
        for idx in selected_indices:
            self.composite_led_sources[idx] = source_label
        self.composite_spectrum_curve.setData(WAVELENGTHS, self.composite_led_values)
        self.update_composite_spectrum_range()
        labels = ", ".join(str(int(WAVELENGTHS[idx])) for idx in selected_indices)
        self.composite_status_label.setText(
            f"Saved channels: {labels} nm at {self.latest_timestamp_ms} ms "
            f"as average of {sample_count} new final graph values."
        )

    def store_composite_calibration_columns(self, selected_indices, sample_values: np.ndarray | None = None) -> None:
        active_profile = self.calibration_window.active_graph_profile()
        if active_profile is None:
            for idx in selected_indices:
                self.composite_dark_values[idx] = np.nan
                self.composite_reference_values[idx] = np.nan
                self.composite_transmission_values[idx] = np.nan
                self.composite_absorbance_values[idx] = np.nan
                self.composite_calibration_profiles[idx] = "no active profile"
            return

        profile_data = self.calibration_window.calibration[active_profile]
        dark = profile_data["dark"]
        reference = profile_data["reference"]
        profile_name = self.calibration_window.profile_display_name(active_profile)
        if dark is None or reference is None:
            for idx in selected_indices:
                self.composite_dark_values[idx] = np.nan
                self.composite_reference_values[idx] = np.nan
                self.composite_transmission_values[idx] = np.nan
                self.composite_absorbance_values[idx] = np.nan
                self.composite_calibration_profiles[idx] = profile_name
            return

        dark = np.array(dark, dtype=float)
        reference = np.array(reference, dtype=float)
        sample = np.array(sample_values if sample_values is not None else self.latest_sensor_values, dtype=float)
        denominator = reference - dark
        valid_denominator = np.isfinite(denominator) & (np.abs(denominator) >= 1e-9)
        transmission = np.full(18, np.nan, dtype=float)
        transmission[valid_denominator] = (sample[valid_denominator] - dark[valid_denominator]) / denominator[valid_denominator]
        absorbance = np.full(18, np.nan, dtype=float)
        valid_transmission = np.isfinite(transmission) & (transmission > 0)
        absorbance[valid_transmission] = -np.log10(transmission[valid_transmission])

        self.composite_dark_values[selected_indices] = dark[selected_indices]
        self.composite_reference_values[selected_indices] = reference[selected_indices]
        self.composite_transmission_values[selected_indices] = transmission[selected_indices]
        self.composite_absorbance_values[selected_indices] = absorbance[selected_indices]
        for idx in selected_indices:
            self.composite_calibration_profiles[idx] = profile_name

    def update_composite_dimensions(self) -> None:
        self.composite_spectrum_plot.setMinimumHeight(self.composite_plot_height_spin.value())
        self.composite_spectrum_plot.setMaximumHeight(self.composite_plot_height_spin.value())
        self.update_composite_spectrum_range()
        self.composite_export_widget.adjustSize()

    def update_composite_spectrum_range(self) -> None:
        all_values = [self.composite_led_values[np.isfinite(self.composite_led_values)]]
        for _name, _curve, values in self.saved_sample_curves:
            all_values.append(values[np.isfinite(values)])
        finite_values = np.concatenate([values for values in all_values if values.size]) if any(values.size for values in all_values) else np.array([])
        max_value = float(np.max(finite_values)) if finite_values.size else 1.0
        y_max = max(max_value * 1.12, 1.0)
        self.composite_spectrum_plot.setYRange(0.0, y_max, padding=0)
        self.composite_spectral_background.setYMax(y_max)
        self.composite_grid_item.setYMax(y_max)

    def refresh_composite_legend(self) -> None:
        self.composite_legend.clear()
        for name, curve, _values in self.saved_sample_curves:
            self.composite_legend.addItem(curve, name)
        self.refresh_sample_remove_combo()

    def refresh_sample_remove_combo(self) -> None:
        if not hasattr(self, "sample_remove_combo"):
            return
        current_index = self.sample_remove_combo.currentIndex()
        self.sample_remove_combo.blockSignals(True)
        self.sample_remove_combo.clear()
        if not self.saved_sample_curves:
            self.sample_remove_combo.addItem("No lines saved", -1)
        else:
            for index, (name, _curve, _values) in enumerate(self.saved_sample_curves):
                self.sample_remove_combo.addItem(name, index)
            if 0 <= current_index < self.sample_remove_combo.count():
                self.sample_remove_combo.setCurrentIndex(current_index)
        self.sample_remove_combo.blockSignals(False)

    def current_composite_export_snapshot(self, sample_name: str) -> dict:
        return {
            "sample_name": sample_name,
            "graph_values": np.array(self.composite_led_values, dtype=float),
            "sparkfun_values": np.array(self.composite_sparkfun_values, dtype=float),
            "raw_values": np.array(self.composite_raw_values, dtype=float),
            "mean_values": np.array(self.composite_mean_values, dtype=float),
            "std_values": np.array(self.composite_std_values, dtype=float),
            "cv_values": np.array(self.composite_cv_values, dtype=float),
            "dark_values": np.array(self.composite_dark_values, dtype=float),
            "reference_values": np.array(self.composite_reference_values, dtype=float),
            "transmission_values": np.array(self.composite_transmission_values, dtype=float),
            "absorbance_values": np.array(self.composite_absorbance_values, dtype=float),
            "led_sources": list(self.composite_led_sources),
            "calibration_profiles": list(self.composite_calibration_profiles),
        }

    def reset_current_composite_arrays(self) -> None:
        self.composite_led_values[:] = np.nan
        self.composite_sparkfun_values[:] = np.nan
        self.composite_raw_values[:] = np.nan
        self.composite_mean_values[:] = np.nan
        self.composite_std_values[:] = np.nan
        self.composite_cv_values[:] = np.nan
        self.composite_dark_values[:] = np.nan
        self.composite_reference_values[:] = np.nan
        self.composite_transmission_values[:] = np.nan
        self.composite_absorbance_values[:] = np.nan
        self.composite_led_sources = [""] * 18
        self.composite_calibration_profiles = [""] * 18

    def add_current_sample_curve(self) -> None:
        values = np.array(self.composite_led_values, dtype=float)
        if not np.isfinite(values).any():
            QtWidgets.QMessageBox.warning(self, "No sample", "Save at least one channel for this sample first.")
            return

        name = self.sample_name_edit.text().strip() or f"Sample {self.saved_sample_counter}"
        color = QtGui.QColor(self.sample_color_combo.currentData())
        curve = self.composite_spectrum_plot.plot(
            WAVELENGTHS,
            values,
            pen=pg.mkPen(color, width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(color),
            symbolPen=pg.mkPen(color),
            name=name,
        )
        self.saved_sample_curves.append((name, curve, values))
        self.saved_sample_exports.append(self.current_composite_export_snapshot(name))
        self.saved_sample_counter += 1
        self.sample_name_edit.setText(f"Sample {self.saved_sample_counter}")
        self.reset_current_composite_arrays()
        self.composite_spectrum_curve.setData(WAVELENGTHS, self.composite_led_values)
        self.refresh_composite_legend()
        self.update_composite_spectrum_range()
        self.composite_status_label.setText(f"Sample added to comparison: {name}. Build the next sample now.")

    def clear_sample_curves(self) -> None:
        for _name, curve, _values in self.saved_sample_curves:
            self.composite_spectrum_plot.removeItem(curve)
        self.saved_sample_curves.clear()
        self.saved_sample_exports.clear()
        self.saved_sample_counter = 1
        self.sample_name_edit.setText("Sample 1")
        self.refresh_composite_legend()
        self.update_composite_spectrum_range()
        self.composite_status_label.setText("Sample comparison cleared.")

    def remove_selected_sample_curve(self) -> None:
        selected_index = self.sample_remove_combo.currentData()
        if selected_index is None or selected_index < 0 or selected_index >= len(self.saved_sample_curves):
            QtWidgets.QMessageBox.warning(self, "No line", "Select a saved line to remove first.")
            return

        name, curve, _values = self.saved_sample_curves.pop(selected_index)
        if selected_index < len(self.saved_sample_exports):
            self.saved_sample_exports.pop(selected_index)
        self.composite_spectrum_plot.removeItem(curve)
        self.refresh_composite_legend()
        self.update_composite_spectrum_range()
        self.composite_status_label.setText(f"Line removed from comparison: {name}.")

    def _selected_composite_source_label(self, selected_indices) -> str:
        selected_set = set(selected_indices)
        if selected_set == set(range(0, 6)):
            return "UV LED"
        if selected_set == set(range(6, 12)):
            return "White LED"
        if selected_set == set(range(12, 18)):
            return "IR LED"
        return "Custom selection"

    def save_composite_spectrum_png(self) -> None:
        raw_name = self.composite_png_name_edit.text().strip()
        if not raw_name:
            QtWidgets.QMessageBox.warning(self, "File name", "Enter a file name first.")
            return

        safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in raw_name)
        if not safe_name.lower().endswith(".png"):
            safe_name += ".png"

        output_path = Path(__file__).resolve().parent / safe_name
        pixmap = self.composite_export_widget.grab()
        if pixmap.save(str(output_path), "PNG"):
            self.composite_status_label.setText(f"PNG saved: {output_path}")
        else:
            QtWidgets.QMessageBox.warning(self, "PNG export", "Failed to save PNG.")

    def save_composite_spectrum_csv(self) -> None:
        raw_name = self.composite_png_name_edit.text().strip()
        if not raw_name:
            QtWidgets.QMessageBox.warning(self, "File name", "Enter a file name first.")
            return

        safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in raw_name)
        if not safe_name.lower().endswith(".csv"):
            safe_name += ".csv"

        snapshots = list(self.saved_sample_exports)
        if not snapshots and np.isfinite(self.composite_led_values).any():
            snapshots.append(self.current_composite_export_snapshot(self.sample_name_edit.text().strip() or "Preview current sample"))
        if not snapshots:
            QtWidgets.QMessageBox.warning(self, "No data", "Save channels first or add at least one sample.")
            return

        output_path = Path(__file__).resolve().parent / safe_name
        headers = [
            "Sample name",
            "LED source",
            "Calibration profile",
            "Wavelength (nm)",
            "Calibrated value SparkFun AS7265x (a.u.)",
            "Raw value AS7265x (counts)",
            "Mean spectral response (a.u.)",
            "Standard deviation spectral response (a.u.)",
            "Coefficient of variation spectral response (%)",
            "Dark profile (a.u.)",
            "Blank/reference profile (a.u.)",
            "Normalised response / transmission (-)",
            "Absorbance A = -log10(transmission) (-)",
            "Value in plotted graph (-)",
        ]

        try:
            with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.writer(csv_file, delimiter=";")
                writer.writerow(headers)
                for snapshot in snapshots:
                    for idx, wavelength in enumerate(WAVELENGTHS):
                        graph_value = snapshot["graph_values"][idx]
                        if not np.isfinite(graph_value):
                            continue
                        writer.writerow([
                            snapshot["sample_name"],
                            snapshot["led_sources"][idx],
                            snapshot["calibration_profiles"][idx],
                            f"{wavelength:.0f}",
                            self._csv_float(snapshot["sparkfun_values"][idx]),
                            self._csv_float(snapshot["raw_values"][idx]),
                            self._csv_float(snapshot["mean_values"][idx]),
                            self._csv_float(snapshot["std_values"][idx]),
                            self._csv_float(snapshot["cv_values"][idx]),
                            self._csv_float(snapshot["dark_values"][idx]),
                            self._csv_float(snapshot["reference_values"][idx]),
                            self._csv_float(snapshot["transmission_values"][idx]),
                            self._csv_float(snapshot["absorbance_values"][idx]),
                            self._csv_float(graph_value),
                        ])
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "CSV export", f"Failed to save CSV: {exc}")
            return

        self.composite_status_label.setText(f"CSV saved: {output_path}")

    def _csv_float(self, value: float) -> str:
        return "" if not np.isfinite(value) else f"{float(value):.8g}"

    def clear_composite_spectrum(self) -> None:
        self.reset_current_composite_arrays()
        self.composite_spectrum_curve.setData(WAVELENGTHS, self.composite_led_values)
        self.update_composite_spectrum_range()
        self.composite_status_label.setText("Composite spectrum cleared.")

    def clear_history(self) -> None:
        self.time_history.clear()
        for history in self.channel_history:
            history.clear()
        for curve in self.curves:
            curve.setData([], [])

    def refresh_stats(self) -> None:
        if self.latest_temperature_c is None:
            temperature_text = "Cuvette temperature: no measurement"
        else:
            temperature_text = (
                f"Cuvette temperature: {self.latest_temperature_c:.3f} deg C "
                f"@ {self.latest_temperature_timestamp_ms} ms"
            )

        self.stats_label.setText(
            f"Samples in buffer: {len(self.time_history)}\n"
            f"Stream FPS: {self.stream_fps:.1f}\n"
            f"Last time: {self.latest_timestamp_ms} ms\n"
            f"Last max value: {float(np.max(self.latest_values)):.3f}\n"
            f"{temperature_text}"
        )

    def append_log(self, text: str) -> None:
        self.log_text.appendPlainText(text)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event) -> None:
        self.reader.disconnect_port()
        if hasattr(self, "led_spectra_page"):
            self.led_spectra_page.close()
        super().closeEvent(event)


def main() -> None:
    global APP_INSTANCE
    pg.setConfigOptions(antialias=True, background="#020617", foreground="#e2e8f0")
    app = QtWidgets.QApplication.instance()
    if app is None:
        APP_INSTANCE = QtWidgets.QApplication(sys.argv)
        app = APP_INSTANCE
    window = As7265xMonitor()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
