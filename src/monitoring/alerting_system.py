"""
Real-time Alerting System for Predictive Maintenance
Handles alert generation, notification delivery, and escalation management.
"""

import json
import time
import smtplib
import boto3
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import logging
import threading
from queue import Queue, PriorityQueue
import requests
import pandas as pd


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertType(Enum):
    """Types of alerts that can be generated."""
    EQUIPMENT_FAILURE = "equipment_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SENSOR_ANOMALY = "sensor_anomaly"
    MODEL_DRIFT = "model_drift"
    SYSTEM_ERROR = "system_error"


@dataclass
class Alert:
    """Data class representing an alert."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    device_id: str
    message: str
    timestamp: datetime
    details: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False
    escalated: bool = False

    def __lt__(self, other):
        """Support priority queue ordering by severity."""
        return self.severity.value > other.severity.value


class AlertingSystem:
    """
    Comprehensive alerting system for predictive maintenance.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the alerting system."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Alert configuration
        self.monitoring_config = config.get('monitoring', {})
        self.alert_config = self.monitoring_config.get('alerts', {})

        # Alert processing
        self.alert_queue = PriorityQueue()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.suppressed_alerts: Dict[str, datetime] = {}

        # Processing thread
        self.is_processing = False
        self.processing_thread = None

        # Notification channels
        self.email_enabled = self.alert_config.get('email_enabled', False)
        self.sms_enabled = self.alert_config.get('sms_enabled', False)
        self.slack_enabled = self.alert_config.get('slack_enabled', False)

        # Initialize notification clients
        self._initialize_notification_clients()

        # Alert rules and thresholds
        self.alert_rules = self._load_alert_rules()

        # Callbacks for custom alert handling
        self.alert_callbacks: List[Callable] = []

    def _initialize_notification_clients(self) -> None:
        """Initialize various notification clients."""
        try:
            # Initialize AWS SNS client
            if 'aws' in self.config:
                self.sns_client = boto3.client(
                    'sns',
                    region_name=self.config['aws']['region']
                )
                self.logger.info("AWS SNS client initialized")

            self.logger.info("Notification clients initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize notification clients: {str(e)}")

    def _load_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load alert rules and thresholds from configuration."""
        default_rules = {
            'equipment_failure_probability': {
                'threshold': 0.8,
                'severity': AlertSeverity.CRITICAL,
                'alert_type': AlertType.EQUIPMENT_FAILURE
            },
            'sensor_anomaly_score': {
                'threshold': 0.9,
                'severity': AlertSeverity.HIGH,
                'alert_type': AlertType.SENSOR_ANOMALY
            },
            'model_performance_drop': {
                'threshold': 0.1,
                'severity': AlertSeverity.MEDIUM,
                'alert_type': AlertType.MODEL_DRIFT
            }
        }

        return default_rules

    def start_processing(self) -> None:
        """Start the alert processing system."""
        if self.is_processing:
            self.logger.warning("Alert processing already started")
            return

        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.logger.info("Alert processing started")

    def stop_processing(self) -> None:
        """Stop the alert processing system."""
        self.is_processing = False

        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)

        self.logger.info("Alert processing stopped")

    def _processing_loop(self) -> None:
        """Main alert processing loop."""
        while self.is_processing:
            try:
                # Process alerts from queue
                if not self.alert_queue.empty():
                    alert = self.alert_queue.get(timeout=1.0)
                    self._process_alert(alert)
                    self.alert_queue.task_done()

                time.sleep(1.0)

            except Exception as e:
                if "timed out" not in str(e).lower():
                    self.logger.error(f"Error in alert processing loop: {str(e)}")

    def create_alert(self, alert_type: AlertType, severity: AlertSeverity,
                    device_id: str, message: str, details: Dict[str, Any] = None) -> Alert:
        """Create a new alert."""
        alert_id = f"{alert_type.value}_{device_id}_{int(time.time())}"

        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            device_id=device_id,
            message=message,
            timestamp=datetime.now(),
            details=details or {}
        )

        # Add to processing queue
        self.alert_queue.put(alert)
        self.logger.info(f"Alert created: {alert_id} - {message}")

        return alert

    def _process_alert(self, alert: Alert) -> None:
        """Process an individual alert."""
        try:
            # Add to active alerts
            self.active_alerts[alert.alert_id] = alert

            # Send notifications
            self._send_notifications(alert)

            # Execute custom callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {str(e)}")

            # Add to history
            self.alert_history.append(alert)

            self.logger.info(f"Alert processed: {alert.alert_id}")

        except Exception as e:
            self.logger.error(f"Failed to process alert {alert.alert_id}: {str(e)}")

    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        try:
            # Send SNS notification
            if hasattr(self, 'sns_client'):
                self._send_sns_notification(alert)

        except Exception as e:
            self.logger.error(f"Failed to send notifications: {str(e)}")

    def _send_sns_notification(self, alert: Alert) -> None:
        """Send SNS notification for an alert."""
        try:
            sns_config = self.config.get('aws', {}).get('sns', {})
            topic_arn = sns_config.get('topic_arn')

            if not topic_arn:
                return

            message = f'''
PREDICTIVE MAINTENANCE ALERT

Severity: {alert.severity.name}
Type: {alert.alert_type.value}
Device: {alert.device_id}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{alert.message}

Alert ID: {alert.alert_id}
'''

            response = self.sns_client.publish(
                TopicArn=topic_arn,
                Message=message,
                Subject=f"[{alert.severity.name}] Equipment Alert: {alert.device_id}"
            )

            self.logger.info(f"SNS notification sent for alert {alert.alert_id}")

        except Exception as e:
            self.logger.error(f"Failed to send SNS notification: {str(e)}")

    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.details['acknowledged_by'] = user
            alert.details['acknowledged_at'] = datetime.now().isoformat()

            self.logger.info(f"Alert acknowledged: {alert_id} by {user}")
            return True

        return False

    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.details['resolved_by'] = user
            alert.details['resolved_at'] = datetime.now().isoformat()

            self.logger.info(f"Alert resolved: {alert_id} by {user}")
            return True

        return False

    def process_sensor_data(self, sensor_data: pd.DataFrame) -> None:
        """Process sensor data and generate alerts based on rules."""
        try:
            for _, row in sensor_data.iterrows():
                device_id = row.get('device_id', 'unknown')

                # Check temperature threshold
                if 'temperature' in row and row['temperature'] > 80:
                    self.create_alert(
                        AlertType.SENSOR_ANOMALY,
                        AlertSeverity.HIGH,
                        device_id,
                        f"High temperature detected: {row['temperature']:.1f}°C",
                        {'temperature': row['temperature'], 'threshold': 80}
                    )

                # Check vibration levels
                if 'vibration' in row and row['vibration'] > 5.0:
                    self.create_alert(
                        AlertType.SENSOR_ANOMALY,
                        AlertSeverity.MEDIUM,
                        device_id,
                        f"High vibration detected: {row['vibration']:.2f} mm/s",
                        {'vibration': row['vibration'], 'threshold': 5.0}
                    )

        except Exception as e:
            self.logger.error(f"Error processing sensor data for alerts: {str(e)}")

    def process_model_prediction(self, device_id: str, failure_probability: float) -> None:
        """Process model prediction and generate alerts if needed."""
        try:
            threshold = self.alert_rules['equipment_failure_probability']['threshold']

            if failure_probability > threshold:
                severity = AlertSeverity.CRITICAL if failure_probability > 0.9 else AlertSeverity.HIGH

                self.create_alert(
                    AlertType.EQUIPMENT_FAILURE,
                    severity,
                    device_id,
                    f"High failure probability detected: {failure_probability:.1%}",
                    {
                        'failure_probability': failure_probability,
                        'threshold': threshold,
                        'recommended_action': 'Schedule immediate maintenance'
                    }
                )

        except Exception as e:
            self.logger.error(f"Error processing model prediction: {str(e)}")

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get statistics about alerts."""
        return {
            'total_alerts': len(self.alert_history),
            'active_alerts': len(self.active_alerts),
            'queue_size': self.alert_queue.qsize()
        }


if __name__ == "__main__":
    # Example usage
    config = {
        'aws': {
            'region': 'us-east-1',
            'sns': {
                'topic_arn': 'arn:aws:sns:us-east-1:123456789:test'
            }
        },
        'monitoring': {
            'alerts': {
                'email_enabled': True,
                'sms_enabled': True
            }
        }
    }

    # Initialize alerting system
    alerting = AlertingSystem(config)
    alerting.start_processing()

    # Create test alert
    alert = alerting.create_alert(
        AlertType.EQUIPMENT_FAILURE,
        AlertSeverity.HIGH,
        "test_device_001",
        "Equipment failure predicted with 85% probability"
    )

    print("Alerting system test completed!")
