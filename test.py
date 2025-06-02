# main.py
import sys
import json
import time
import threading
import asyncio
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from abc import ABC, abstractmethod

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QPushButton, QSpinBox, QGroupBox, QComboBox,
    QCheckBox, QTabWidget, QSplitter, QFrame, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QMutex, QMutexLocker, QMetaObject
from PyQt5.QtGui import QPixmap, QFont
import pyqtgraph as pg
import numpy as np
import cv2

# Attempt to import Pupil Labs API
try:
    from pupil_labs.realtime_api.simple import Device, discover_one_device
    from pupil_labs.realtime_api.models import GazeDatum, VideoFrame
    PUPIL_LABS_API_AVAILABLE = True
except ImportError:
    PUPIL_LABS_API_AVAILABLE = False
    # Define dummy classes if API is not available, so the rest of the type hints don't break
    class GazeDatum: pass
    class VideoFrame: pass
    class Device: pass


# ======================= LOGGING SETUP =======================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ======================= DATA MODELS =======================

@dataclass
class EyeTrackingData:
    """Data model for eye tracking information"""
    x: float = 0.0  # Normalized gaze x (0-1)
    y: float = 0.0  # Normalized gaze y (0-1)
    worn: bool = False
    pupil_diameter_left: float = 0.0
    eyeball_center_left_x: float = 0.0
    eyeball_center_left_y: float = 0.0
    eyeball_center_left_z: float = 0.0
    optical_axis_left_x: float = 0.0
    optical_axis_left_y: float = 0.0
    optical_axis_left_z: float = 0.0
    pupil_diameter_right: float = 0.0
    eyeball_center_right_x: float = 0.0
    eyeball_center_right_y: float = 0.0
    eyeball_center_right_z: float = 0.0
    optical_axis_right_x: float = 0.0
    optical_axis_right_y: float = 0.0
    optical_axis_right_z: float = 0.0
    eyelid_angle_top_left: float = 0.0
    eyelid_angle_bottom_left: float = 0.0
    eyelid_aperture_left: float = 0.0
    eyelid_angle_top_right: float = 0.0
    eyelid_angle_bottom_right: float = 0.0
    eyelid_aperture_right: float = 0.0
    timestamp_unix_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ======================= DATA INTERFACES =======================

class IDataProvider(ABC):
    """Interface for data providers"""
    @abstractmethod
    def start_streaming(self) -> None:
        pass
    
    @abstractmethod
    def stop_streaming(self) -> None:
        pass
    
    @abstractmethod
    def register_callback(self, callback: Callable[[EyeTrackingData], None]) -> None:
        pass


class IVideoProvider(ABC):
    """Interface for video providers"""
    @abstractmethod
    def get_eye_frame(self) -> Optional[np.ndarray]:
        pass
    
    @abstractmethod
    def get_scene_frame(self) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def start_streaming(self) -> None: # Added for consistency in managing async tasks
        pass

    @abstractmethod
    def stop_streaming(self) -> None: # Added for consistency
        pass


# ======================= DATA PROCESSING =======================

class DataBuffer:
    """Thread-safe circular buffer for eye tracking data"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.data = deque(maxlen=max_size)
        self.mutex = QMutex()
    
    def add_data(self, data: EyeTrackingData) -> None:
        with QMutexLocker(self.mutex):
            self.data.append(data)
    
    def get_recent_data(self, seconds: float) -> List[EyeTrackingData]:
        """Get data from the last N seconds"""
        with QMutexLocker(self.mutex):
            if not self.data:
                return []
            
            # Use the timestamp from the data itself, not time.time()
            # as data might be delayed or replayed.
            # Assuming data is added chronologically.
            if not self.data:
                return []
            
            latest_timestamp = self.data[-1].timestamp_unix_seconds
            cutoff_time = latest_timestamp - seconds
            
            recent_data = []
            # Iterate from the end of the deque
            for item in reversed(self.data):
                if item.timestamp_unix_seconds >= cutoff_time:
                    recent_data.append(item)
                else:
                    # Data is older than the cutoff, and since deque is ordered, we can stop
                    break 
            return list(reversed(recent_data)) # Return in chronological order
    
    def get_all_data(self) -> List[EyeTrackingData]:
        with QMutexLocker(self.mutex):
            return list(self.data)


class DataProcessor:
    """Processes and analyzes eye tracking data"""
    
    def __init__(self, buffer: DataBuffer):
        self.buffer = buffer
    
    def get_pupil_diameter_series(self, seconds: float, eye: str = 'left') -> tuple:
        """Get time series data for pupil diameter"""
        data = self.buffer.get_recent_data(seconds)
        if not data:
            return [], []
        
        timestamps = []
        diameters = []
        
        field_name = f'pupil_diameter_{eye}'
        
        for item in data:
            timestamps.append(item.timestamp_unix_seconds)
            diameters.append(getattr(item, field_name))
        
        return timestamps, diameters
    
    def get_gaze_position_series(self, seconds: float) -> tuple:
        """Get time series data for gaze position (normalized)"""
        data = self.buffer.get_recent_data(seconds)
        if not data:
            return [], []
        
        timestamps = [] # For x-axis of chart
        x_positions = []
        y_positions = []
        
        for item in data:
            timestamps.append(item.timestamp_unix_seconds)
            x_positions.append(item.x)
            y_positions.append(item.y)
        
        return timestamps, x_positions, y_positions # Return timestamps as well
    
    def get_eyelid_aperture_series(self, seconds: float, eye: str = 'left') -> tuple:
        """Get time series data for eyelid aperture"""
        data = self.buffer.get_recent_data(seconds)
        if not data:
            return [], []

        timestamps = []
        apertures = []
        field_name = f'eyelid_aperture_{eye}'

        for item in data:
            timestamps.append(item.timestamp_unix_seconds)
            apertures.append(getattr(item, field_name))
        
        return timestamps, apertures

    def get_latest_stats(self) -> Dict[str, Any]:
        """Get latest statistics"""
        # Get data from the last 0.5 seconds to find the most recent valid point
        recent_data_list = self.buffer.get_recent_data(0.5) 
        if not recent_data_list:
            # Return a dictionary with default/empty values if no recent data
            return {
                'pupil_left': 0.0, 'pupil_right': 0.0,
                'gaze_x': 0.0, 'gaze_y': 0.0, 'worn': False,
                'eyelid_aperture_left': 0.0, 'eyelid_aperture_right': 0.0,
                'timestamp': 0.0
            }
        
        latest = recent_data_list[-1] # The most recent item
        return {
            'pupil_left': latest.pupil_diameter_left,
            'pupil_right': latest.pupil_diameter_right,
            'gaze_x': latest.x, # Normalized
            'gaze_y': latest.y, # Normalized
            'worn': latest.worn,
            'eyelid_aperture_left': latest.eyelid_aperture_left,
            'eyelid_aperture_right': latest.eyelid_aperture_right,
            'timestamp': latest.timestamp_unix_seconds
        }


# ======================= DATA PROVIDERS =======================

class PupilLabsDataProvider(IDataProvider):
    """Data provider using Pupil Labs Realtime API for gaze data."""
    def __init__(self, device: Device, loop: asyncio.AbstractEventLoop):
        self.device = device
        self.loop = loop
        self.callbacks: List[Callable[[EyeTrackingData], None]] = []
        self._is_streaming = False
        self.gaze_task: Optional[asyncio.Task] = None

    def register_callback(self, callback: Callable[[EyeTrackingData], None]) -> None:
        self.callbacks.append(callback)

    def start_streaming(self) -> None:
        if not self.device:
            logging.warning("PupilLabsDataProvider: No device, cannot start streaming.")
            return
        if self._is_streaming and self.gaze_task and not self.gaze_task.done():
            logging.info("PupilLabsDataProvider: Already streaming gaze data.")
            return
        
        self._is_streaming = True
        logging.info("PupilLabsDataProvider: Starting gaze data streaming task.")
        # Ensure task is created and run on the provided asyncio loop
        self.gaze_task = self.loop.create_task(self._stream_gaze_data_loop())


    async def _stream_gaze_data_loop(self):
        logging.info("PupilLabsDataProvider: Gaze data streaming loop started.")
        try:
            while self._is_streaming:
                if not self.device or not self.device.is_connected: # Check device status
                    logging.warning("PupilLabsDataProvider: Device disconnected or not available. Stopping gaze stream.")
                    self._is_streaming = False
                    break
                try:
                    gaze_datum: GazeDatum = await self.device.receive_gaze_datum(timeout_seconds=1.0)
                    if gaze_datum:
                        parsed_data = self._parse_pupil_data(gaze_datum)
                        for callback in self.callbacks:
                            callback(parsed_data)
                except asyncio.TimeoutError:
                    logging.debug("PupilLabsDataProvider: Timeout receiving gaze datum.")
                    if not self._is_streaming: break # Exit if streaming was stopped during timeout
                    continue # Continue loop if still streaming
                except Exception as e:
                    logging.error(f"PupilLabsDataProvider: Error receiving gaze data: {e}")
                    # Consider stopping or attempting to re-establish if specific errors occur
                    await asyncio.sleep(0.1) # Brief pause before retrying or exiting
            
        except asyncio.CancelledError:
            logging.info("PupilLabsDataProvider: Gaze data streaming task cancelled.")
        except Exception as e:
            logging.error(f"PupilLabsDataProvider: Unhandled exception in gaze streaming loop: {e}")
        finally:
            self._is_streaming = False
            logging.info("PupilLabsDataProvider: Gaze data streaming loop finished.")

    def stop_streaming(self) -> None:
        logging.info("PupilLabsDataProvider: Stopping gaze data streaming.")
        self._is_streaming = False
        if self.gaze_task and not self.gaze_task.done():
            # Request cancellation of the task
            self.loop.call_soon_threadsafe(self.gaze_task.cancel)
        self.gaze_task = None


    def _parse_pupil_data(self, datum: GazeDatum) -> EyeTrackingData:
        # Initialize with defaults from EyeTrackingData
        data = EyeTrackingData(timestamp_unix_seconds=datum.timestamp_unix_seconds)

        data.x = datum.x if datum.x is not None else 0.0
        data.y = datum.y if datum.y is not None else 0.0
        data.worn = datum.worn if datum.worn is not None else False

        if datum.left_eye_data:
            se_left = datum.left_eye_data
            data.pupil_diameter_left = se_left.pupil_diameter_mm if se_left.pupil_diameter_mm is not None else 0.0
            if se_left.eyeball_center_mm:
                data.eyeball_center_left_x = se_left.eyeball_center_mm[0]
                data.eyeball_center_left_y = se_left.eyeball_center_mm[1]
                data.eyeball_center_left_z = se_left.eyeball_center_mm[2]
            if se_left.optical_axis_normalized:
                data.optical_axis_left_x = se_left.optical_axis_normalized[0]
                data.optical_axis_left_y = se_left.optical_axis_normalized[1]
                data.optical_axis_left_z = se_left.optical_axis_normalized[2]

        if datum.right_eye_data:
            se_right = datum.right_eye_data
            data.pupil_diameter_right = se_right.pupil_diameter_mm if se_right.pupil_diameter_mm is not None else 0.0
            if se_right.eyeball_center_mm:
                data.eyeball_center_right_x = se_right.eyeball_center_mm[0]
                data.eyeball_center_right_y = se_right.eyeball_center_mm[1]
                data.eyeball_center_right_z = se_right.eyeball_center_mm[2]
            if se_right.optical_axis_normalized:
                data.optical_axis_right_x = se_right.optical_axis_normalized[0]
                data.optical_axis_right_y = se_right.optical_axis_normalized[1]
                data.optical_axis_right_z = se_right.optical_axis_normalized[2]

        if datum.eyestate_left:
            es_left = datum.eyestate_left
            data.eyelid_angle_top_left = es_left.eyelid_angle_top_degrees if es_left.eyelid_angle_top_degrees is not None else 0.0
            data.eyelid_angle_bottom_left = es_left.eyelid_angle_bottom_degrees if es_left.eyelid_angle_bottom_degrees is not None else 0.0
            data.eyelid_aperture_left = es_left.eyelid_aperture_mm if es_left.eyelid_aperture_mm is not None else 0.0

        if datum.eyestate_right:
            es_right = datum.eyestate_right
            data.eyelid_angle_top_right = es_right.eyelid_angle_top_degrees if es_right.eyelid_angle_top_degrees is not None else 0.0
            data.eyelid_angle_bottom_right = es_right.eyelid_angle_bottom_degrees if es_right.eyelid_angle_bottom_degrees is not None else 0.0
            data.eyelid_aperture_right = es_right.eyelid_aperture_mm if es_right.eyelid_aperture_mm is not None else 0.0
        
        return data


class PupilLabsVideoProvider(IVideoProvider):
    """Video provider using Pupil Labs Realtime API for eye and scene cameras."""
    def __init__(self, device: Device, loop: asyncio.AbstractEventLoop):
        self.device = device
        self.loop = loop
        self._is_streaming = False
        self.eye_video_task: Optional[asyncio.Task] = None
        self.scene_video_task: Optional[asyncio.Task] = None

        self.latest_eye_frame: Optional[np.ndarray] = None
        self.latest_scene_frame: Optional[np.ndarray] = None
        self.frame_mutex = QMutex() # To protect access to latest_eye_frame and latest_scene_frame

    def start_streaming(self) -> None:
        if not self.device:
            logging.warning("PupilLabsVideoProvider: No device, cannot start streaming.")
            return
        if self._is_streaming and \
           (self.eye_video_task and not self.eye_video_task.done()) and \
           (self.scene_video_task and not self.scene_video_task.done()):
            logging.info("PupilLabsVideoProvider: Already streaming video.")
            return

        self._is_streaming = True
        logging.info("PupilLabsVideoProvider: Starting video streaming tasks.")
        self.eye_video_task = self.loop.create_task(self._stream_eye_video_loop())
        self.scene_video_task = self.loop.create_task(self._stream_scene_video_loop())

    async def _stream_eye_video_loop(self):
        logging.info("PupilLabsVideoProvider: Eye video streaming loop started.")
        try:
            while self._is_streaming:
                if not self.device or not self.device.is_connected:
                    logging.warning("PupilLabsVideoProvider: Device disconnected. Stopping eye video stream.")
                    self._is_streaming = False; break
                try:
                    video_frame: VideoFrame = await self.device.receive_eyes_video_frame(timeout_seconds=1.0)
                    if video_frame and video_frame.bgr_pixels is not None:
                        with QMutexLocker(self.frame_mutex):
                            self.latest_eye_frame = video_frame.bgr_pixels.copy() # BGR format
                except asyncio.TimeoutError:
                    logging.debug("PupilLabsVideoProvider: Timeout receiving eye video frame.")
                    if not self._is_streaming: break
                    continue
                except Exception as e:
                    logging.error(f"PupilLabsVideoProvider: Error receiving eye video: {e}")
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logging.info("PupilLabsVideoProvider: Eye video streaming task cancelled.")
        except Exception as e:
            logging.error(f"PupilLabsVideoProvider: Unhandled exception in eye video loop: {e}")
        finally:
            self._is_streaming = False # Ensure flag is reset
            logging.info("PupilLabsVideoProvider: Eye video streaming loop finished.")
            with QMutexLocker(self.frame_mutex): # Clear frame on stop
                self.latest_eye_frame = None


    async def _stream_scene_video_loop(self):
        logging.info("PupilLabsVideoProvider: Scene video streaming loop started.")
        try:
            while self._is_streaming:
                if not self.device or not self.device.is_connected:
                    logging.warning("PupilLabsVideoProvider: Device disconnected. Stopping scene video stream.")
                    self._is_streaming = False; break
                try:
                    video_frame: VideoFrame = await self.device.receive_scene_video_frame(timeout_seconds=1.0)
                    if video_frame and video_frame.bgr_pixels is not None:
                        with QMutexLocker(self.frame_mutex):
                            self.latest_scene_frame = video_frame.bgr_pixels.copy() # BGR format
                except asyncio.TimeoutError:
                    logging.debug("PupilLabsVideoProvider: Timeout receiving scene video frame.")
                    if not self._is_streaming: break
                    continue
                except Exception as e:
                    logging.error(f"PupilLabsVideoProvider: Error receiving scene video: {e}")
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logging.info("PupilLabsVideoProvider: Scene video streaming task cancelled.")
        except Exception as e:
            logging.error(f"PupilLabsVideoProvider: Unhandled exception in scene video loop: {e}")
        finally:
            self._is_streaming = False # Ensure flag is reset
            logging.info("PupilLabsVideoProvider: Scene video streaming loop finished.")
            with QMutexLocker(self.frame_mutex): # Clear frame on stop
                self.latest_scene_frame = None


    def stop_streaming(self) -> None:
        logging.info("PupilLabsVideoProvider: Stopping video streaming.")
        self._is_streaming = False # Signal loops to stop
        if self.eye_video_task and not self.eye_video_task.done():
            self.loop.call_soon_threadsafe(self.eye_video_task.cancel)
        if self.scene_video_task and not self.scene_video_task.done():
            self.loop.call_soon_threadsafe(self.scene_video_task.cancel)
        self.eye_video_task = None
        self.scene_video_task = None


    def get_eye_frame(self) -> Optional[np.ndarray]:
        with QMutexLocker(self.frame_mutex):
            return self.latest_eye_frame.copy() if self.latest_eye_frame is not None else None

    def get_scene_frame(self) -> Optional[np.ndarray]:
        with QMutexLocker(self.frame_mutex):
            return self.latest_scene_frame.copy() if self.latest_scene_frame is not None else None


class MockDataProvider(IDataProvider):
    """Mock data provider for testing when Pupil Labs API is not available or device not connected."""
    def __init__(self):
        self._is_streaming = False
        self.callbacks: List[Callable[[EyeTrackingData], None]] = []
        self.thread: Optional[threading.Thread] = None
        logging.info("MockDataProvider initialized.")

    def register_callback(self, callback: Callable[[EyeTrackingData], None]) -> None:
        self.callbacks.append(callback)

    def start_streaming(self) -> None:
        if self._is_streaming:
            return
        logging.info("MockDataProvider: Starting data generation.")
        self._is_streaming = True
        self.thread = threading.Thread(target=self._generate_data, daemon=True)
        self.thread.start()

    def stop_streaming(self) -> None:
        logging.info("MockDataProvider: Stopping data generation.")
        self._is_streaming = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None
    
    def _generate_data(self) -> None:
        """Generate mock eye tracking data at ~200Hz"""
        while self._is_streaming:
            current_time = time.time()
            noise = np.random.normal(0, 0.01) # Smaller noise for normalized gaze
            
            data = EyeTrackingData(
                x=0.5 + 0.2 * np.sin(current_time * 0.5) + noise, # Normalized (0-1)
                y=0.5 + 0.1 * np.cos(current_time * 0.3) + noise, # Normalized (0-1)
                worn=True,
                pupil_diameter_left=3.5 + 0.5 * np.sin(current_time * 2) + noise * 10,
                pupil_diameter_right=3.4 + 0.5 * np.sin(current_time * 2 + 0.1) + noise * 10,
                eyelid_aperture_left=8.0 + 1.0 * abs(np.sin(current_time * 0.8)),
                eyelid_aperture_right=7.8 + 1.0 * abs(np.cos(current_time * 0.8)),
                timestamp_unix_seconds=current_time
            )
            # Ensure x, y are within [0, 1]
            data.x = np.clip(data.x, 0, 1)
            data.y = np.clip(data.y, 0, 1)

            for callback in self.callbacks:
                callback(data)
            
            time.sleep(1/200) # 200Hz


class MockVideoProvider(IVideoProvider):
    """Mock video provider for testing"""
    def __init__(self):
        self.eye_frame = self._generate_mock_frame((200, 100), "Eye View (Mock)")
        self.scene_frame = self._generate_mock_frame((640, 480), "Scene View (Mock)") # Typical scene cam size
        logging.info("MockVideoProvider initialized.")

    def _generate_mock_frame(self, size: tuple, text: str) -> np.ndarray:
        frame = np.random.randint(50, 150, (*size, 3), dtype=np.uint8) # Darker background
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        # Add a moving circle to simulate activity
        center_x = int(size[1] / 2 + (size[1] / 4) * np.sin(time.time()))
        center_y = int(size[0] / 2 + (size[0] / 4) * np.cos(time.time()))
        cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
        return frame

    def get_eye_frame(self) -> Optional[np.ndarray]:
        return self._generate_mock_frame((200, 100), f"Eye Mock {time.time():.1f}")
    
    def get_scene_frame(self) -> Optional[np.ndarray]:
        return self._generate_mock_frame((640, 480), f"Scene Mock {time.time():.1f}")

    def start_streaming(self) -> None: # Mock provider doesn't need async tasks
        logging.info("MockVideoProvider: Start streaming (no-op).")
        pass

    def stop_streaming(self) -> None: # Mock provider doesn't need async tasks
        logging.info("MockVideoProvider: Stop streaming (no-op).")
        pass


# ======================= UI COMPONENTS =======================

class VideoWidget(QLabel):
    """Widget for displaying video streams"""
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.setMinimumSize(320, 240) # Adjusted for typical video aspect ratios
        self.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.setAlignment(Qt.AlignCenter)
        self.setText(f"{title}\nNo Signal")
        self.setScaledContents(True) # Important for proper scaling
    
    def update_frame(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            self.setText(f"{self.title}\nNo Signal") # Reset if frame is None
            return
        
        height, width, channel = frame.shape
        bytes_per_line = channel * width # Usually 3 for BGR/RGB
        
        # Convert BGR (OpenCV default) to RGB for QImage
        if channel == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_image_format = pg.QtGui.QImage.Format_RGB888
        elif channel == 1: # Grayscale
            rgb_frame = frame # No conversion needed, but QImage format changes
            q_image_format = pg.QtGui.QImage.Format_Grayscale8
        else: # Unsupported format
            logging.warning(f"VideoWidget: Unsupported frame channel count: {channel}")
            self.setText(f"{self.title}\nUnsupported Format")
            return

        q_image = pg.QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, q_image_format)
        
        # QPixmap must be created from a QImage that is not going out of scope.
        # Making a copy of q_image can help if there are issues with data pointers.
        self.setPixmap(QPixmap.fromImage(q_image.copy()))


class ChartWidget(QWidget):
    """Widget for displaying real-time charts"""
    def __init__(self, title: str, y_label: str = "Value"):
        super().__init__()
        self.title = title
        layout = QVBoxLayout()
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', y_label)
        self.plot_widget.setLabel('bottom', 'Time (s, relative to now)')
        self.plot_widget.showGrid(True, True)
        self.plot_widget.setBackground('w') # White background
        
        self.curves = {}
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
    
    def add_curve(self, name: str, color: str, width: int = 2) -> None:
        pen = pg.mkPen(color=color, width=width)
        self.curves[name] = self.plot_widget.plot([], [], pen=pen, name=name)
    
    def update_curve(self, name: str, timestamps: List[float], y_data: List[float]) -> None:
        if name in self.curves and timestamps and y_data:
            if len(timestamps) != len(y_data):
                logging.warning(f"ChartWidget '{self.title}': Mismatch in timestamps and y_data length for curve '{name}'.")
                return

            # Convert absolute timestamps to relative time (seconds ago from latest timestamp)
            if timestamps:
                latest_timestamp = timestamps[-1] # Assuming timestamps are sorted
                relative_times = [ts - latest_timestamp for ts in timestamps]
                self.curves[name].setData(relative_times, y_data)


class StatsWidget(QWidget):
    """Widget for displaying current statistics"""
    def __init__(self):
        super().__init__()
        self.stats_labels: Dict[str, QLabel] = {}
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        title = QLabel("Current Stats")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        self.stats_frame = QFrame()
        self.stats_layout = QGridLayout()
        self.stats_frame.setLayout(self.stats_layout)
        layout.addWidget(self.stats_frame)
        
        self.setLayout(layout)
        self.setup_stats_labels()
    
    def setup_stats_labels(self):
        stats_info = [
            ('Pupil Left', 'pupil_left', 'mm'),
            ('Pupil Right', 'pupil_right', 'mm'),
            ('Gaze X (Norm)', 'gaze_x', ''), # Normalized
            ('Gaze Y (Norm)', 'gaze_y', ''), # Normalized
            ('Eyelid Apert. L', 'eyelid_aperture_left', 'mm'),
            ('Eyelid Apert. R', 'eyelid_aperture_right', 'mm'),
            ('Device Worn', 'worn', ''),
            ('Timestamp', 'timestamp', 's')
        ]
        
        for i, (label_text, key, unit) in enumerate(stats_info):
            name_label = QLabel(f"{label_text}:")
            value_label = QLabel("--")
            value_label.setStyleSheet("font-weight: bold; color: blue;")
            
            self.stats_layout.addWidget(name_label, i, 0)
            self.stats_layout.addWidget(value_label, i, 1)
            
            if unit:
                unit_label = QLabel(unit)
                self.stats_layout.addWidget(unit_label, i, 2)
            
            self.stats_labels[key] = value_label
    
    def update_stats(self, stats_data: Dict[str, Any]):
        for key, value in stats_data.items():
            if key in self.stats_labels:
                if isinstance(value, bool):
                    display_value = "Yes" if value else "No"
                elif isinstance(value, float):
                    if key == 'timestamp':
                         display_value = f"{value:.2f}" # Keep more precision for timestamp
                    else:
                         display_value = f"{value:.3f}" # More precision for normalized gaze
                else:
                    display_value = str(value)
                
                self.stats_labels[key].setText(display_value)


class ControlPanel(QWidget):
    """Control panel for application settings"""
    time_window_changed = pyqtSignal(int)
    # chart_type_changed = pyqtSignal(str) # Chart type selection removed, all charts shown in tabs
    recording_toggled = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        time_group = QGroupBox("Time Window")
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Seconds:"))
        self.time_spinbox = QSpinBox()
        self.time_spinbox.setRange(1, 600) # Increased max time window
        self.time_spinbox.setValue(10)
        self.time_spinbox.valueChanged.connect(self.time_window_changed.emit)
        time_layout.addWidget(self.time_spinbox)
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        # Chart type combo box removed as tabs are used for charts
        
        record_group = QGroupBox("Recording (Placeholder)")
        record_layout = QVBoxLayout()
        self.record_button = QPushButton("Start Recording")
        self.record_button.setCheckable(True)
        self.record_button.toggled.connect(self._on_record_toggled)
        record_layout.addWidget(self.record_button)
        record_group.setLayout(record_layout)
        layout.addWidget(record_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def _on_record_toggled(self, checked: bool):
        self.record_button.setText("Stop Recording" if checked else "Start Recording")
        self.recording_toggled.emit(checked)
        logging.info(f"Recording toggled: {'ON' if checked else 'OFF'}")


# ======================= MAIN APPLICATION =======================

class EyeTrackingApp(QMainWindow):
    """Main application window"""
    # Signal to indicate device status
    device_status_signal = pyqtSignal(str) 

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Tracking Data Visualization (Pupil Labs Integration)")
        self.setGeometry(50, 50, 1600, 950) # Adjusted size
        
        self.data_buffer = DataBuffer()
        self.data_processor = DataProcessor(self.data_buffer)
        
        # Async setup
        self.async_loop: Optional[asyncio.AbstractEventLoop] = None
        self.async_thread: Optional[threading.Thread] = None
        self.pupil_device: Optional[Device] = None

        self.data_provider: Optional[IDataProvider] = None
        self.video_provider: Optional[IVideoProvider] = None
        
        self.setup_ui()
        self.setup_timers()
        self.setup_connections()
        
        self.time_window = self.control_panel.time_spinbox.value() # Initial value from control panel
        self.is_recording = False

        # Start async operations for device connection and data streaming
        self._start_async_infrastructure()

        self.device_status_signal.connect(self.show_device_status_message)


    def _start_async_infrastructure(self):
        if not PUPIL_LABS_API_AVAILABLE:
            logging.warning("Pupil Labs API not found. Falling back to Mock Providers.")
            self.device_status_signal.emit("Pupil Labs API not found. Using Mock Data.")
            self._use_mock_providers()
            # Start streaming for mock providers
            if self.data_provider: self.data_provider.start_streaming()
            if self.video_provider: self.video_provider.start_streaming() # Though mock video is on-demand
            return

        self.async_loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self._run_async_loop_forever, daemon=True)
        self.async_thread.start()

        # Schedule the device initialization on the new loop
        asyncio.run_coroutine_threadsafe(self._initialize_device_and_providers(), self.async_loop)


    def _run_async_loop_forever(self):
        if not self.async_loop: return
        logging.info("Asyncio event loop starting in dedicated thread.")
        asyncio.set_event_loop(self.async_loop)
        try:
            self.async_loop.run_forever()
        finally:
            # This part runs when loop.stop() is called
            if self.async_loop.is_running(): # Ensure it's stopped before closing
                 self.async_loop.call_soon_threadsafe(self.async_loop.stop)
            # self.async_loop.close() # Close loop after it has stopped
            logging.info("Asyncio event loop has stopped.")


    async def _initialize_device_and_providers(self):
        """Async method to discover device and set up providers."""
        if not self.async_loop: return

        try:
            logging.info("Attempting to discover Pupil Labs device...")
            self.device_status_signal.emit("Discovering Pupil Labs device...")
            # discover_one_device is synchronous, but Device() can be used
            # We need to manage start/stop manually if not using `async with`
            self.pupil_device = Device() # Uses discover_one_device internally
            await self.pupil_device.start(wait_for_calibration=True, timeout_calibration_seconds=10) # Start and wait for calibration
            
            status_msg = f"Connected to: {self.pupil_device.phone_name} (Serial: {self.pupil_device.module_serial})"
            logging.info(status_msg)
            self.device_status_signal.emit(status_msg)

            self.data_provider = PupilLabsDataProvider(self.pupil_device, self.async_loop)
            self.video_provider = PupilLabsVideoProvider(self.pupil_device, self.async_loop)
            
            self.data_provider.register_callback(self.on_new_data_received)
            
            self.data_provider.start_streaming()
            self.video_provider.start_streaming()

        except TimeoutError:
            logging.error("Could not find a Pupil Labs device or calibration timed out.")
            self.device_status_signal.emit("Pupil Labs device not found or calibration failed. Using Mock Data.")
            self.pupil_device = None
            self._use_mock_providers()
            if self.data_provider: self.data_provider.start_streaming()
            if self.video_provider: self.video_provider.start_streaming()
        except Exception as e:
            logging.error(f"Error initializing Pupil Labs device: {e}")
            self.device_status_signal.emit(f"Error: {e}. Using Mock Data.")
            self.pupil_device = None
            self._use_mock_providers()
            if self.data_provider: self.data_provider.start_streaming()
            if self.video_provider: self.video_provider.start_streaming()

    def _use_mock_providers(self):
        """Sets up mock providers if real device fails or API is unavailable."""
        logging.info("Setting up MockDataProvider and MockVideoProvider.")
        self.data_provider = MockDataProvider()
        self.video_provider = MockVideoProvider()
        self.data_provider.register_callback(self.on_new_data_received)
        # Mock providers might need explicit start if they have internal threads
        # self.data_provider.start_streaming()
        # self.video_provider.start_streaming() # MockVideoProvider start is no-op

    def show_device_status_message(self, message: str):
        """Shows a non-blocking message, e.g., in status bar or a temporary label."""
        # For simplicity, using a QMessageBox here. For a real app, a status bar is better.
        # QMessageBox.information(self, "Device Status", message)
        # Or, update a status label if you add one to the UI
        if hasattr(self, 'status_bar_label'):
             self.status_bar_label.setText(message)
        else: # Fallback to console
            logging.info(f"UI Status: {message}")


    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel, 0) # Stretch factor 0
        
        center_splitter = QSplitter(Qt.Vertical)
        
        video_widget_container = QWidget()
        video_layout = QHBoxLayout()
        self.eye_video = VideoWidget("Eye Camera")
        self.scene_video = VideoWidget("Scene Camera")
        video_layout.addWidget(self.eye_video)
        video_layout.addWidget(self.scene_video)
        video_widget_container.setLayout(video_layout)
        center_splitter.addWidget(video_widget_container)
        
        chart_tabs = QTabWidget()
        self.pupil_chart = ChartWidget("Pupil Diameter Over Time", "Diameter (mm)")
        self.pupil_chart.add_curve("Left Eye", "blue")
        self.pupil_chart.add_curve("Right Eye", "red")
        chart_tabs.addTab(self.pupil_chart, "Pupil Diameter")
        
        self.gaze_chart = ChartWidget("Gaze Position Over Time", "Normalized Position") # Label updated
        self.gaze_chart.add_curve("X Position", "green")
        self.gaze_chart.add_curve("Y Position", "orange")
        chart_tabs.addTab(self.gaze_chart, "Gaze Position")
        
        self.eyelid_chart = ChartWidget("Eyelid Aperture Over Time", "Aperture (mm)")
        self.eyelid_chart.add_curve("Left Eye", "purple")
        self.eyelid_chart.add_curve("Right Eye", "brown")
        chart_tabs.addTab(self.eyelid_chart, "Eyelid Aperture")
        
        center_splitter.addWidget(chart_tabs)
        center_splitter.setSizes([int(self.height() * 0.35), int(self.height() * 0.65)]) # Adjusted splitter sizes
        
        main_layout.addWidget(center_splitter, 2) # Stretch factor 2
        
        self.stats_widget = StatsWidget()
        main_layout.addWidget(self.stats_widget, 0) # Stretch factor 0
        
        central_widget.setLayout(main_layout)

        # Add a status bar
        self.status_bar = self.statusBar()
        self.status_bar_label = QLabel("Initializing...")
        self.status_bar.addWidget(self.status_bar_label)
    
    def setup_timers(self):
        self.chart_timer = QTimer()
        self.chart_timer.timeout.connect(self.update_charts_and_stats) # Combined update
        self.chart_timer.start(33)  # ~30 FPS for charts and stats

        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_videos)
        self.video_timer.start(66)  # ~15 FPS for video
    
    def setup_connections(self):
        self.control_panel.time_window_changed.connect(self.on_time_window_changed)
        self.control_panel.recording_toggled.connect(self.on_recording_toggled)
    
    def on_new_data_received(self, data: EyeTrackingData):
        """Callback for new data from the provider (called from provider's thread or async context)."""
        self.data_buffer.add_data(data) # DataBuffer is thread-safe
    
    def on_time_window_changed(self, seconds: int):
        self.time_window = seconds
        logging.info(f"Chart time window changed to: {seconds}s")
    
    def on_recording_toggled(self, recording: bool):
        self.is_recording = recording
        # Actual recording logic would go here (e.g., writing data_buffer to file)
        logging.info(f"Recording {'started' if recording else 'stopped'}.")
    
    def update_charts_and_stats(self):
        """Update all charts and statistics display."""
        if not self.data_provider: return # Don't update if no provider

        # Update charts
        times_pupil_l, diams_l = self.data_processor.get_pupil_diameter_series(self.time_window, 'left')
        times_pupil_r, diams_r = self.data_processor.get_pupil_diameter_series(self.time_window, 'right')
        self.pupil_chart.update_curve("Left Eye", times_pupil_l, diams_l)
        self.pupil_chart.update_curve("Right Eye", times_pupil_r, diams_r)
        
        times_gaze, gaze_x, gaze_y = self.data_processor.get_gaze_position_series(self.time_window)
        self.gaze_chart.update_curve("X Position", times_gaze, gaze_x)
        self.gaze_chart.update_curve("Y Position", times_gaze, gaze_y)

        times_eyelid_l, apertures_l = self.data_processor.get_eyelid_aperture_series(self.time_window, 'left')
        times_eyelid_r, apertures_r = self.data_processor.get_eyelid_aperture_series(self.time_window, 'right')
        self.eyelid_chart.update_curve("Left Eye", times_eyelid_l, apertures_l)
        self.eyelid_chart.update_curve("Right Eye", times_eyelid_r, apertures_r)

        # Update stats
        current_stats = self.data_processor.get_latest_stats()
        self.stats_widget.update_stats(current_stats)

    def update_videos(self):
        if not self.video_provider: return # Don't update if no provider

        eye_frame = self.video_provider.get_eye_frame()
        scene_frame = self.video_provider.get_scene_frame()
        
        self.eye_video.update_frame(eye_frame)
        self.scene_video.update_frame(scene_frame)
            
    def closeEvent(self, event):
        """Handle application close: shut down async tasks and device."""
        logging.info("Closing application...")

        # Stop providers first (signals async loops to wind down)
        if self.data_provider:
            self.data_provider.stop_streaming()
        if self.video_provider:
            self.video_provider.stop_streaming()

        # If Pupil Labs device was used, close it
        if self.pupil_device and self.async_loop and self.pupil_device.is_connected:
            logging.info("Closing Pupil Labs device connection...")
            try:
                # Ensure close is called from the loop it runs on or is thread-safe
                future = asyncio.run_coroutine_threadsafe(self.pupil_device.close(), self.async_loop)
                future.result(timeout=5) # Wait for close to complete
                logging.info("Pupil Labs device closed.")
            except Exception as e:
                logging.error(f"Error closing Pupil Labs device: {e}")
        
        # Stop the asyncio event loop
        if self.async_loop and self.async_loop.is_running():
            logging.info("Stopping asyncio event loop...")
            self.async_loop.call_soon_threadsafe(self.async_loop.stop)
        
        # Wait for the async thread to finish
        if self.async_thread and self.async_thread.is_alive():
            logging.info("Joining asyncio thread...")
            self.async_thread.join(timeout=5) # Wait for thread to join
            if self.async_thread.is_alive():
                logging.warning("Asyncio thread did not terminate cleanly.")
        
        # Close the loop if it hasn't been closed by run_forever finishing
        if self.async_loop and not self.async_loop.is_closed():
             # This should be done after run_forever finishes.
             # If loop was started with run_forever, it closes itself on stop.
             # If it was managed differently, loop.close() might be needed here.
             # For safety, ensure it's stopped.
             if not self.async_loop.is_running(): # Should be stopped by now
                 self.async_loop.close()
                 logging.info("Asyncio event loop closed.")


        logging.info("Application cleanup finished.")
        super().closeEvent(event)


# ======================= APPLICATION ENTRY POINT =======================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    if not PUPIL_LABS_API_AVAILABLE:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText("Pupil Labs Realtime API not found.")
        msg_box.setInformativeText("Please install 'pupil-labs-realtime-api' (`pip install pupil-labs-realtime-api`). The application will run with mock data.")
        msg_box.setWindowTitle("API Missing")
        msg_box.exec_()

    window = EyeTrackingApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

# ======================= REQUIREMENTS =======================
"""
Required packages:
pip install PyQt5 pyqtgraph numpy opencv-python pupil-labs-realtime-api

- PyQt5: For the GUI.
- pyqtgraph: For plotting.
- numpy: For numerical operations, especially with video frames.
- opencv-python: For video frame manipulation (BGR to RGB conversion).
- pupil-labs-realtime-api: For connecting to Pupil Labs eye trackers.
"""