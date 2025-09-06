"""
AWS utilities for the predictive maintenance system.
"""

import boto3
import json
import time
from typing import Dict, Any, List, Optional
from botocore.exceptions import ClientError
import logging


class AWSUtils:
    """Utility class for AWS operations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize AWS utilities with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.region = config['aws']['region']

        # Initialize clients
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.sns_client = boto3.client('sns', region_name=self.region)
        self.iot_client = boto3.client('iot', region_name=self.region)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=self.region)

    def create_s3_bucket_if_not_exists(self, bucket_name: str) -> bool:
        """
        Create S3 bucket if it doesn't exist.

        Args:
            bucket_name: Name of the bucket to create

        Returns:
            True if bucket exists or was created successfully
        """
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.logger.info(f"S3 bucket {bucket_name} already exists")
            return True

        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                # Bucket doesn't exist, create it
                try:
                    if self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )

                    self.logger.info(f"S3 bucket {bucket_name} created successfully")
                    return True

                except ClientError as create_error:
                    self.logger.error(f"Failed to create S3 bucket {bucket_name}: {create_error}")
                    return False
            else:
                self.logger.error(f"Error accessing S3 bucket {bucket_name}: {e}")
                return False

    def upload_to_s3(self, local_file: str, bucket: str, s3_key: str) -> bool:
        """
        Upload file to S3.

        Args:
            local_file: Local file path
            bucket: S3 bucket name
            s3_key: S3 object key

        Returns:
            True if successful
        """
        try:
            self.s3_client.upload_file(local_file, bucket, s3_key)
            self.logger.info(f"File uploaded to s3://{bucket}/{s3_key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to upload file to S3: {e}")
            return False

    def create_sns_topic_if_not_exists(self, topic_name: str) -> Optional[str]:
        """
        Create SNS topic if it doesn't exist.

        Args:
            topic_name: Name of the topic to create

        Returns:
            Topic ARN if successful, None otherwise
        """
        try:
            response = self.sns_client.create_topic(Name=topic_name)
            topic_arn = response['TopicArn']
            self.logger.info(f"SNS topic created/verified: {topic_arn}")
            return topic_arn

        except Exception as e:
            self.logger.error(f"Failed to create SNS topic: {e}")
            return None

    def subscribe_email_to_sns(self, topic_arn: str, email: str) -> bool:
        """
        Subscribe email to SNS topic.

        Args:
            topic_arn: ARN of the SNS topic
            email: Email address to subscribe

        Returns:
            True if successful
        """
        try:
            response = self.sns_client.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint=email
            )
            self.logger.info(f"Email {email} subscribed to SNS topic")
            return True

        except Exception as e:
            self.logger.error(f"Failed to subscribe email to SNS: {e}")
            return False

    def create_iot_thing_if_not_exists(self, thing_name: str) -> bool:
        """
        Create IoT thing if it doesn't exist.

        Args:
            thing_name: Name of the IoT thing

        Returns:
            True if successful
        """
        try:
            self.iot_client.describe_thing(thingName=thing_name)
            self.logger.info(f"IoT thing {thing_name} already exists")
            return True

        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                try:
                    self.iot_client.create_thing(thingName=thing_name)
                    self.logger.info(f"IoT thing {thing_name} created")
                    return True
                except Exception as create_error:
                    self.logger.error(f"Failed to create IoT thing: {create_error}")
                    return False
            else:
                self.logger.error(f"Error checking IoT thing: {e}")
                return False

    def put_cloudwatch_metric(self, namespace: str, metric_name: str, 
                             value: float, unit: str = 'Count',
                             dimensions: List[Dict[str, str]] = None) -> bool:
        """
        Put metric data to CloudWatch.

        Args:
            namespace: CloudWatch namespace
            metric_name: Name of the metric
            value: Metric value
            unit: Metric unit
            dimensions: Metric dimensions

        Returns:
            True if successful
        """
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': time.time()
            }

            if dimensions:
                metric_data['Dimensions'] = dimensions

            self.cloudwatch_client.put_metric_data(
                Namespace=namespace,
                MetricData=[metric_data]
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to put CloudWatch metric: {e}")
            return False

    def setup_aws_resources(self) -> Dict[str, Any]:
        """
        Set up required AWS resources for the predictive maintenance system.

        Returns:
            Dictionary with resource information
        """
        resources = {}

        # Create S3 bucket
        bucket_name = self.config['aws']['s3']['bucket']
        if self.create_s3_bucket_if_not_exists(bucket_name):
            resources['s3_bucket'] = bucket_name

        # Create SNS topic
        topic_arn = self.create_sns_topic_if_not_exists('predictive-maintenance-alerts')
        if topic_arn:
            resources['sns_topic_arn'] = topic_arn

            # Subscribe emails
            subscribers = self.config.get('aws', {}).get('sns', {}).get('email_subscribers', [])
            for email in subscribers:
                self.subscribe_email_to_sns(topic_arn, email)

        # Create IoT things for test devices
        for i in range(1, 6):
            thing_name = f"predictive-maintenance-device-{i:03d}"
            if self.create_iot_thing_if_not_exists(thing_name):
                resources[f'iot_thing_{i}'] = thing_name

        self.logger.info("AWS resources setup completed")
        return resources
