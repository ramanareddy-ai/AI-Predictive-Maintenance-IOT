"""
Data Streaming Module for Real-time Predictive Maintenance
Handles real-time data streaming and processing for IoT sensor data.
"""

import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional, Callable
from queue import Queue
import threading
import time

from kafka import KafkaProducer, KafkaConsumer
import boto3
from awsiot import mqtt_connection_builder

from ..utils.logger import setup_logger


class DataStreamer:
    """
    Real-time data streaming handler for predictive maintenance.

    Supports multiple streaming backends:
    - AWS Kinesis
    - Apache Kafka
    - AWS IoT Core
    - WebSockets
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data streamer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logger(__name__, config.get('logging', {}))

        # Streaming configuration
        self.streaming_enabled = True
        self.batch_size = config.get('sensors', {}).get('data_collection', {}).get('batch_size', 100)

        # Initialize streaming clients
        self.kafka_producer = None
        self.kinesis_client = None
        self.iot_client = None

        # Data processing queue
        self.processing_queue = Queue()
        self.processing_thread = None
        self.is_processing = False

        # Callbacks for real-time processing
        self.stream_callbacks: List[Callable] = []

        self._initialize_streaming_clients()

    def _initialize_streaming_clients(self) -> None:
        """Initialize various streaming clients based on configuration."""
        try:
            # Initialize Kafka producer
            self._initialize_kafka()

            # Initialize AWS Kinesis client
            self._initialize_kinesis()

            # Initialize AWS IoT client
            self._initialize_aws_iot()

            self.logger.info("Streaming clients initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize streaming clients: {str(e)}")

    def _initialize_kafka(self) -> None:
        """Initialize Kafka producer."""
        try:
            # Basic Kafka configuration
            kafka_config = {
                'bootstrap_servers': ['localhost:9092'],  # Default local Kafka
                'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
                'key_serializer': lambda k: str(k).encode('utf-8') if k else None
            }

            self.kafka_producer = KafkaProducer(**kafka_config)
            self.logger.info("Kafka producer initialized")

        except Exception as e:
            self.logger.warning(f"Failed to initialize Kafka producer: {str(e)}")

    def _initialize_kinesis(self) -> None:
        """Initialize AWS Kinesis client."""
        try:
            if 'aws' in self.config:
                self.kinesis_client = boto3.client(
                    'kinesis',
                    region_name=self.config['aws']['region']
                )
                self.logger.info("AWS Kinesis client initialized")

        except Exception as e:
            self.logger.warning(f"Failed to initialize Kinesis client: {str(e)}")

    def _initialize_aws_iot(self) -> None:
        """Initialize AWS IoT client."""
        try:
            if 'aws' in self.config and 'iot' in self.config['aws']:
                iot_config = self.config['aws']['iot']

                self.iot_client = mqtt_connection_builder.mtls_from_path(
                    endpoint=iot_config['endpoint'],
                    port=iot_config['port'],
                    cert_filepath=iot_config['cert_file'],
                    pri_key_filepath=iot_config['key_file'],
                    ca_filepath=iot_config['ca_file'],
                    client_id=f"data-streamer-{int(time.time())}",
                    clean_session=False,
                    keep_alive_secs=30
                )

                self.logger.info("AWS IoT client initialized")

        except Exception as e:
            self.logger.warning(f"Failed to initialize AWS IoT client: {str(e)}")

    def start_streaming(self) -> None:
        """Start the data streaming process."""
        if self.is_processing:
            self.logger.warning("Streaming already started")
            return

        self.is_processing = True

        # Connect AWS IoT if available
        if self.iot_client:
            try:
                connect_future = self.iot_client.connect()
                connect_future.result()
                self.logger.info("Connected to AWS IoT for streaming")
            except Exception as e:
                self.logger.warning(f"Failed to connect to AWS IoT: {str(e)}")

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.logger.info("Data streaming started")

    def stop_streaming(self) -> None:
        """Stop the data streaming process."""
        self.is_processing = False

        # Disconnect AWS IoT
        if self.iot_client:
            try:
                disconnect_future = self.iot_client.disconnect()
                disconnect_future.result()
            except Exception as e:
                self.logger.warning(f"Failed to disconnect from AWS IoT: {str(e)}")

        # Close Kafka producer
        if self.kafka_producer:
            self.kafka_producer.close()

        # Wait for processing thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)

        self.logger.info("Data streaming stopped")

    def stream_data(self, data: pd.DataFrame) -> None:
        """
        Stream data to configured endpoints.

        Args:
            data: DataFrame containing sensor data to stream
        """
        if not self.streaming_enabled:
            return

        try:
            # Add to processing queue
            self.processing_queue.put(data)

        except Exception as e:
            self.logger.error(f"Failed to queue data for streaming: {str(e)}")

    def _processing_loop(self) -> None:
        """Main processing loop for streaming data."""
        while self.is_processing:
            try:
                # Get data from queue with timeout
                data = self.processing_queue.get(timeout=1.0)

                # Stream to different endpoints
                self._stream_to_kafka(data)
                self._stream_to_kinesis(data)
                self._stream_to_iot_core(data)

                # Execute callbacks
                for callback in self.stream_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in stream callback: {str(e)}")

                self.processing_queue.task_done()

            except Exception as e:
                if "timed out" not in str(e).lower():
                    self.logger.error(f"Error in processing loop: {str(e)}")

    def _stream_to_kafka(self, data: pd.DataFrame) -> None:
        """Stream data to Kafka topic."""
        if not self.kafka_producer:
            return

        try:
            topic = "predictive-maintenance-sensor-data"

            for _, row in data.iterrows():
                message = {
                    'timestamp': row['timestamp'].isoformat() if pd.notnull(row['timestamp']) else datetime.now().isoformat(),
                    'device_id': row.get('device_id', 'unknown'),
                    'sensor_data': row.drop(['timestamp', 'device_id']).to_dict()
                }

                self.kafka_producer.send(
                    topic,
                    key=row.get('device_id', 'unknown'),
                    value=message
                )

            self.kafka_producer.flush()
            self.logger.debug(f"Streamed {len(data)} records to Kafka")

        except Exception as e:
            self.logger.error(f"Failed to stream to Kafka: {str(e)}")

    def _stream_to_kinesis(self, data: pd.DataFrame) -> None:
        """Stream data to AWS Kinesis."""
        if not self.kinesis_client:
            return

        try:
            stream_name = "predictive-maintenance-stream"

            records = []
            for _, row in data.iterrows():
                record = {
                    'Data': json.dumps({
                        'timestamp': row['timestamp'].isoformat() if pd.notnull(row['timestamp']) else datetime.now().isoformat(),
                        'device_id': row.get('device_id', 'unknown'),
                        'sensor_data': row.drop(['timestamp', 'device_id']).to_dict()
                    }),
                    'PartitionKey': row.get('device_id', 'unknown')
                }
                records.append(record)

            # Send in batches of 500 (Kinesis limit)
            batch_size = 500
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]

                response = self.kinesis_client.put_records(
                    Records=batch,
                    StreamName=stream_name
                )

                # Check for failed records
                if response.get('FailedRecordCount', 0) > 0:
                    self.logger.warning(f"Failed to send {response['FailedRecordCount']} records to Kinesis")

            self.logger.debug(f"Streamed {len(data)} records to Kinesis")

        except Exception as e:
            self.logger.error(f"Failed to stream to Kinesis: {str(e)}")

    def _stream_to_iot_core(self, data: pd.DataFrame) -> None:
        """Stream data to AWS IoT Core."""
        if not self.iot_client:
            return

        try:
            topic = self.config['aws']['iot']['topics']['sensor_data']

            for _, row in data.iterrows():
                message = {
                    'timestamp': row['timestamp'].isoformat() if pd.notnull(row['timestamp']) else datetime.now().isoformat(),
                    'device_id': row.get('device_id', 'unknown'),
                    'sensor_data': row.drop(['timestamp', 'device_id']).to_dict()
                }

                publish_future, _ = self.iot_client.publish(
                    topic=topic,
                    payload=json.dumps(message),
                    qos=1  # At least once delivery
                )

                # Wait for publish to complete (non-blocking)
                # publish_future.result(timeout=1.0)

            self.logger.debug(f"Streamed {len(data)} records to IoT Core")

        except Exception as e:
            self.logger.error(f"Failed to stream to IoT Core: {str(e)}")

    def add_stream_callback(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """Add a callback for real-time stream processing."""
        self.stream_callbacks.append(callback)

    def remove_stream_callback(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """Remove a stream processing callback."""
        if callback in self.stream_callbacks:
            self.stream_callbacks.remove(callback)

    def create_kinesis_stream(self, stream_name: str, shard_count: int = 1) -> bool:
        """
        Create a new Kinesis stream.

        Args:
            stream_name: Name of the stream to create
            shard_count: Number of shards for the stream

        Returns:
            True if successful, False otherwise
        """
        if not self.kinesis_client:
            self.logger.error("Kinesis client not available")
            return False

        try:
            response = self.kinesis_client.create_stream(
                StreamName=stream_name,
                ShardCount=shard_count
            )

            self.logger.info(f"Kinesis stream '{stream_name}' creation initiated")

            # Wait for stream to become active
            waiter = self.kinesis_client.get_waiter('stream_exists')
            waiter.wait(StreamName=stream_name)

            self.logger.info(f"Kinesis stream '{stream_name}' is now active")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create Kinesis stream: {str(e)}")
            return False

    def get_stream_metrics(self) -> Dict[str, Any]:
        """
        Get streaming metrics and status.

        Returns:
            Dictionary containing streaming metrics
        """
        metrics = {
            'queue_size': self.processing_queue.qsize(),
            'streaming_enabled': self.streaming_enabled,
            'is_processing': self.is_processing,
            'kafka_available': self.kafka_producer is not None,
            'kinesis_available': self.kinesis_client is not None,
            'iot_available': self.iot_client is not None,
            'active_callbacks': len(self.stream_callbacks)
        }

        return metrics


class StreamConsumer:
    """Consumer for processing streamed data from various sources."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the stream consumer."""
        self.config = config
        self.logger = setup_logger(__name__, config.get('logging', {}))

        self.kafka_consumer = None
        self.kinesis_client = None
        self.is_consuming = False
        self.consumption_thread = None

        # Processing callbacks
        self.message_callbacks: List[Callable] = []

    def start_kafka_consumer(self, topic: str, group_id: str = "predictive-maintenance") -> None:
        """Start consuming from Kafka topic."""
        try:
            self.kafka_consumer = KafkaConsumer(
                topic,
                bootstrap_servers=['localhost:9092'],
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest'
            )

            self.is_consuming = True
            self.consumption_thread = threading.Thread(target=self._kafka_consumption_loop)
            self.consumption_thread.daemon = True
            self.consumption_thread.start()

            self.logger.info(f"Started Kafka consumer for topic: {topic}")

        except Exception as e:
            self.logger.error(f"Failed to start Kafka consumer: {str(e)}")

    def _kafka_consumption_loop(self) -> None:
        """Kafka message consumption loop."""
        while self.is_consuming and self.kafka_consumer:
            try:
                messages = self.kafka_consumer.poll(timeout_ms=1000)

                for topic_partition, msgs in messages.items():
                    for message in msgs:
                        # Process message
                        self._process_message(message.value)

            except Exception as e:
                self.logger.error(f"Error in Kafka consumption loop: {str(e)}")

    def _process_message(self, message: Dict[str, Any]) -> None:
        """Process a consumed message."""
        try:
            # Execute callbacks
            for callback in self.message_callbacks:
                callback(message)

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")

    def add_message_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback for processing consumed messages."""
        self.message_callbacks.append(callback)

    def stop_consuming(self) -> None:
        """Stop consuming messages."""
        self.is_consuming = False

        if self.kafka_consumer:
            self.kafka_consumer.close()

        if self.consumption_thread:
            self.consumption_thread.join(timeout=5.0)

        self.logger.info("Stream consumption stopped")


if __name__ == "__main__":
    # Example usage
    import yaml

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize streamer
    streamer = DataStreamer(config)
    streamer.start_streaming()

    # Create sample data
    sample_data = pd.DataFrame({
        'timestamp': [datetime.now()],
        'device_id': ['test_device'],
        'temperature': [45.2],
        'vibration': [2.1],
        'pressure': [50.5],
        'rotation_speed': [3500],
        'torque': [78.5],
        'tool_wear': [15.2],
        'power_consumption': [102.3],
        'operating_hours': [150.5]
    })

    # Stream sample data
    streamer.stream_data(sample_data)

    # Wait a bit then stop
    time.sleep(2)
    streamer.stop_streaming()

    print("Data streaming test completed!")
