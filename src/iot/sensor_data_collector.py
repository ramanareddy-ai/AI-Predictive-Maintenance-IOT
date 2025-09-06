"""
IoT Sensor Data Collector for Predictive Maintenance
Handles real-time collection and streaming of sensor data from IoT devices.
"""

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import threading
from queue import Queue, Full
import paho.mqtt.client as mqtt
from awsiot import mqtt_connection_builder
import boto3

from ..utils.logger import setup_logger
from .data_streamer import DataStreamer


@dataclass
class SensorReading:
    """Data class for individual sensor readings."""
    device_id: str
    timestamp: datetime
    temperature: float
    vibration: float
    pressure: float
    rotation_speed: float
    torque: float
    tool_wear: float
    power_consumption: float
    operating_hours: float


class SensorDataCollector:
    """
    Real-time IoT sensor data collector for predictive maintenance.

    Collects sensor data from multiple IoT devices and streams it for processing.
    Supports both MQTT and AWS IoT Core protocols.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sensor data collector.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logger(__name__, config.get('logging', {}))

        # Configuration
        self.collection_frequency = config['sensors']['data_collection']['frequency']
        self.batch_size = config['sensors']['data_collection']['batch_size']
        self.buffer_size = config['sensors']['data_collection']['buffer_size']

        # Data buffer and processing
        self.data_buffer = Queue(maxsize=self.buffer_size)
        self.processing_callbacks: List[Callable] = []

        # MQTT configuration
        self.mqtt_client = None
        self.aws_iot_client = None
        self.is_collecting = False
        self.collection_thread = None

        # Sensor specifications
        self.sensor_specs = config['sensors']

        # Data streamer for real-time processing
        self.data_streamer = DataStreamer(config)

        # Initialize connections
        self._initialize_connections()

    def _initialize_connections(self) -> None:
        """Initialize MQTT and AWS IoT connections."""
        try:
            # Initialize standard MQTT client
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

            # Initialize AWS IoT client if configured
            if 'aws' in self.config and 'iot' in self.config['aws']:
                self._initialize_aws_iot()

            self.logger.info("MQTT connections initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize connections: {str(e)}")
            raise

    def _initialize_aws_iot(self) -> None:
        """Initialize AWS IoT Core connection."""
        try:
            iot_config = self.config['aws']['iot']

            # Build MQTT connection for AWS IoT
            self.aws_iot_client = mqtt_connection_builder.mtls_from_path(
                endpoint=iot_config['endpoint'],
                port=iot_config['port'],
                cert_filepath=iot_config['cert_file'],
                pri_key_filepath=iot_config['key_file'],
                ca_filepath=iot_config['ca_file'],
                client_id=f"sensor-collector-{int(time.time())}",
                clean_session=False,
                keep_alive_secs=30
            )

            self.logger.info("AWS IoT connection initialized")

        except Exception as e:
            self.logger.warning(f"Failed to initialize AWS IoT: {str(e)}")

    def _on_mqtt_connect(self, client, userdata, flags, rc) -> None:
        """MQTT connection callback."""
        if rc == 0:
            self.logger.info("Connected to MQTT broker")
            # Subscribe to sensor data topics
            topics = self.config['aws']['iot']['topics']
            client.subscribe(topics['sensor_data'])
        else:
            self.logger.error(f"Failed to connect to MQTT broker: {rc}")

    def _on_mqtt_message(self, client, userdata, msg) -> None:
        """MQTT message callback."""
        try:
            # Parse incoming sensor data
            data = json.loads(msg.payload.decode())
            sensor_reading = self._parse_sensor_data(data)

            # Add to buffer
            self._add_to_buffer(sensor_reading)

        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {str(e)}")

    def _on_mqtt_disconnect(self, client, userdata, rc) -> None:
        """MQTT disconnect callback."""
        self.logger.warning(f"Disconnected from MQTT broker: {rc}")

    def _parse_sensor_data(self, data: Dict[str, Any]) -> SensorReading:
        """
        Parse raw sensor data into SensorReading object.

        Args:
            data: Raw sensor data dictionary

        Returns:
            Parsed SensorReading object
        """
        return SensorReading(
            device_id=data.get('device_id', 'unknown'),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            temperature=float(data.get('temperature', 0.0)),
            vibration=float(data.get('vibration', 0.0)),
            pressure=float(data.get('pressure', 0.0)),
            rotation_speed=float(data.get('rotation_speed', 0.0)),
            torque=float(data.get('torque', 0.0)),
            tool_wear=float(data.get('tool_wear', 0.0)),
            power_consumption=float(data.get('power_consumption', 0.0)),
            operating_hours=float(data.get('operating_hours', 0.0))
        )

    def _add_to_buffer(self, sensor_reading: SensorReading) -> None:
        """
        Add sensor reading to the processing buffer.

        Args:
            sensor_reading: SensorReading object to add
        """
        try:
            self.data_buffer.put(sensor_reading, block=False)

        except Full:
            # Buffer is full, remove oldest item and add new one
            try:
                self.data_buffer.get(block=False)
                self.data_buffer.put(sensor_reading, block=False)
                self.logger.warning("Data buffer full, dropping oldest reading")
            except:
                self.logger.error("Failed to manage buffer overflow")

    def start_collection(self) -> None:
        """Start collecting sensor data."""
        if self.is_collecting:
            self.logger.warning("Data collection already started")
            return

        self.is_collecting = True

        # Start MQTT connection
        if self.mqtt_client:
            try:
                # Connect to MQTT broker (configure broker details in config)
                self.mqtt_client.connect("localhost", 1883, 60)  # Default local MQTT
                self.mqtt_client.loop_start()

            except Exception as e:
                self.logger.warning(f"Failed to connect to MQTT broker: {str(e)}")

        # Start AWS IoT connection
        if self.aws_iot_client:
            try:
                connect_future = self.aws_iot_client.connect()
                connect_future.result()
                self.logger.info("Connected to AWS IoT Core")

            except Exception as e:
                self.logger.warning(f"Failed to connect to AWS IoT: {str(e)}")

        # Start data collection thread
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        self.logger.info("Sensor data collection started")

    def stop_collection(self) -> None:
        """Stop collecting sensor data."""
        self.is_collecting = False

        # Disconnect MQTT clients
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()

        if self.aws_iot_client:
            disconnect_future = self.aws_iot_client.disconnect()
            disconnect_future.result()

        # Wait for collection thread to finish
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)

        self.logger.info("Sensor data collection stopped")

    def _collection_loop(self) -> None:
        """Main data collection loop."""
        batch = []

        while self.is_collecting:
            try:
                # Collect data from buffer
                if not self.data_buffer.empty():
                    sensor_reading = self.data_buffer.get(timeout=1.0)
                    batch.append(sensor_reading)

                    # Process batch when full
                    if len(batch) >= self.batch_size:
                        self._process_batch(batch)
                        batch = []

                # Generate synthetic data if no real sensors available
                if self.data_buffer.empty() and len(batch) == 0:
                    synthetic_reading = self._generate_synthetic_data()
                    batch.append(synthetic_reading)

                    if len(batch) >= self.batch_size:
                        self._process_batch(batch)
                        batch = []

                time.sleep(self.collection_frequency)

            except Exception as e:
                self.logger.error(f"Error in collection loop: {str(e)}")
                time.sleep(1.0)

        # Process remaining batch
        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch: List[SensorReading]) -> None:
        """
        Process a batch of sensor readings.

        Args:
            batch: List of SensorReading objects
        """
        try:
            # Convert to DataFrame for processing
            df = self._readings_to_dataframe(batch)

            # Apply data validation
            df = self._validate_sensor_data(df)

            # Stream data for real-time processing
            self.data_streamer.stream_data(df)

            # Execute callbacks
            for callback in self.processing_callbacks:
                try:
                    callback(df)
                except Exception as e:
                    self.logger.error(f"Error in processing callback: {str(e)}")

            self.logger.debug(f"Processed batch of {len(batch)} sensor readings")

        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")

    def _readings_to_dataframe(self, readings: List[SensorReading]) -> pd.DataFrame:
        """Convert list of sensor readings to DataFrame."""
        data = []
        for reading in readings:
            data.append({
                'device_id': reading.device_id,
                'timestamp': reading.timestamp,
                'temperature': reading.temperature,
                'vibration': reading.vibration,
                'pressure': reading.pressure,
                'rotation_speed': reading.rotation_speed,
                'torque': reading.torque,
                'tool_wear': reading.tool_wear,
                'power_consumption': reading.power_consumption,
                'operating_hours': reading.operating_hours
            })

        return pd.DataFrame(data)

    def _validate_sensor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate sensor data against specifications.

        Args:
            df: DataFrame with sensor data

        Returns:
            Validated DataFrame
        """
        # Check ranges for each sensor type
        sensor_ranges = {
            'temperature': self.sensor_specs['temperature']['range'],
            'vibration': self.sensor_specs['vibration']['range'],
            'pressure': self.sensor_specs['pressure']['range'],
            'rotation_speed': self.sensor_specs['rotation_speed']['range'],
            'torque': self.sensor_specs['torque']['range']
        }

        for sensor, (min_val, max_val) in sensor_ranges.items():
            if sensor in df.columns:
                # Flag out-of-range values
                out_of_range = (df[sensor] < min_val) | (df[sensor] > max_val)
                if out_of_range.any():
                    self.logger.warning(f"Out-of-range values detected for {sensor}")
                    # Clip values to valid range
                    df[sensor] = df[sensor].clip(min_val, max_val)

        return df

    def _generate_synthetic_data(self) -> SensorReading:
        """Generate synthetic sensor data for testing."""
        now = datetime.now(timezone.utc)

        # Generate realistic sensor values with some variation
        base_temp = 50 + 10 * np.sin(time.time() / 3600)  # Hourly variation
        temperature = base_temp + np.random.normal(0, 2)

        vibration = 2.0 + np.random.exponential(0.5)
        pressure = 45 + np.random.normal(0, 2)
        rotation_speed = 3500 + np.random.normal(0, 100)
        torque = 80 + np.random.normal(0, 5)
        tool_wear = np.random.uniform(0, 100)
        power_consumption = 100 + rotation_speed / 50 + np.random.normal(0, 5)
        operating_hours = (time.time() % 86400) / 3600  # Hours since start of day

        return SensorReading(
            device_id=f"device_{np.random.randint(1, 6)}",
            timestamp=now,
            temperature=temperature,
            vibration=vibration,
            pressure=pressure,
            rotation_speed=rotation_speed,
            torque=torque,
            tool_wear=tool_wear,
            power_consumption=power_consumption,
            operating_hours=operating_hours
        )

    def add_processing_callback(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """Add a callback function for processing collected data."""
        self.processing_callbacks.append(callback)

    def remove_processing_callback(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """Remove a processing callback."""
        if callback in self.processing_callbacks:
            self.processing_callbacks.remove(callback)

    def get_buffer_status(self) -> Dict[str, int]:
        """Get current buffer status."""
        return {
            'buffer_size': self.data_buffer.qsize(),
            'buffer_capacity': self.buffer_size,
            'buffer_utilization': (self.data_buffer.qsize() / self.buffer_size) * 100
        }

    def publish_to_aws_iot(self, topic: str, data: Dict[str, Any]) -> None:
        """Publish data to AWS IoT Core topic."""
        if self.aws_iot_client:
            try:
                publish_future, _ = self.aws_iot_client.publish(
                    topic=topic,
                    payload=json.dumps(data),
                    qos=mqtt.QoS.AT_LEAST_ONCE
                )
                publish_future.result()

            except Exception as e:
                self.logger.error(f"Failed to publish to AWS IoT: {str(e)}")


class MockSensorDevice:
    """Mock IoT sensor device for testing purposes."""

    def __init__(self, device_id: str, mqtt_client: mqtt.Client):
        self.device_id = device_id
        self.mqtt_client = mqtt_client
        self.is_running = False
        self.thread = None

    def start_publishing(self, topic: str, interval: float = 1.0) -> None:
        """Start publishing mock sensor data."""
        self.is_running = True
        self.thread = threading.Thread(
            target=self._publishing_loop, 
            args=(topic, interval)
        )
        self.thread.daemon = True
        self.thread.start()

    def stop_publishing(self) -> None:
        """Stop publishing mock sensor data."""
        self.is_running = False
        if self.thread:
            self.thread.join()

    def _publishing_loop(self, topic: str, interval: float) -> None:
        """Publishing loop for mock data."""
        while self.is_running:
            data = {
                'device_id': self.device_id,
                'timestamp': datetime.now().isoformat(),
                'temperature': 20 + np.random.normal(0, 5),
                'vibration': 1 + np.random.exponential(0.5),
                'pressure': 50 + np.random.normal(0, 3),
                'rotation_speed': 3000 + np.random.normal(0, 100),
                'torque': 75 + np.random.normal(0, 5),
                'tool_wear': np.random.uniform(0, 100),
                'power_consumption': 95 + np.random.normal(0, 10),
                'operating_hours': (time.time() % 86400) / 3600
            }

            self.mqtt_client.publish(topic, json.dumps(data))
            time.sleep(interval)


if __name__ == "__main__":
    # Example usage
    import yaml

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize collector
    collector = SensorDataCollector(config)

    # Add processing callback
    def process_data(df):
        print(f"Processed {len(df)} sensor readings")
        print(df.head())

    collector.add_processing_callback(process_data)

    # Start collection
    collector.start_collection()

    try:
        # Run for 30 seconds
        time.sleep(30)
    finally:
        collector.stop_collection()

    print("Sensor data collector test completed!")
