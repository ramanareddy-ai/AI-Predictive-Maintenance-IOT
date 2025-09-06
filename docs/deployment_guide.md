# Deployment Guide

This guide walks you through deploying the AI-Driven Predictive Maintenance system to AWS.

## Prerequisites

### AWS Account Setup
1. AWS Account with appropriate permissions
2. AWS CLI installed and configured
3. IAM role for SageMaker with necessary permissions

### Local Development Environment
```bash
# Clone the repository
git clone https://github.com/your-username/ai-predictive-maintenance-iot.git
cd ai-predictive-maintenance-iot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Configuration

### 1. Update Configuration File
Edit `config.yaml` with your AWS settings:

```yaml
aws:
  region: "your-aws-region"
  sagemaker:
    role_arn: "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole"
  s3:
    bucket: "your-unique-bucket-name"
  sns:
    topic_arn: "arn:aws:sns:your-region:YOUR_ACCOUNT:predictive-maintenance-alerts"
    email_subscribers:
      - "your-email@company.com"
```

### 2. Set Environment Variables
```bash
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
export SAGEMAKER_ROLE=arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole
export S3_BUCKET=your-unique-bucket-name
```

## Step-by-Step Deployment

### Step 1: Set Up AWS Resources
```bash
# Create S3 bucket and other resources
python scripts/setup_aws_resources.py --config config.yaml
```

### Step 2: Train the Model
```bash
# Train with synthetic data
python scripts/train_model.py --config config.yaml --verbose

# Or train with your own data
python scripts/train_model.py --config config.yaml --data path/to/your/data.csv
```

### Step 3: Deploy Model to SageMaker
```bash
# Deploy the latest trained model
python scripts/deploy_model.py --config config.yaml --test-endpoint

# Or deploy a specific model
python scripts/deploy_model.py --config config.yaml --model-path data/models/your_model.h5
```

### Step 4: Start IoT Data Collection
```python
# In Python script or notebook
from src.iot.sensor_data_collector import SensorDataCollector
from src.utils.config_loader import load_config

config = load_config('config.yaml')
collector = SensorDataCollector(config)
collector.start_collection()
```

## Docker Deployment

### Build and Run Locally
```bash
# Build the Docker image
docker build -t predictive-maintenance .

# Run the container
docker run -it --rm \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/data:/app/data \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  predictive-maintenance \
  python scripts/train_model.py --config config.yaml
```

### Docker Compose for Development
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f predictive-maintenance

# Stop services
docker-compose down
```

## AWS Infrastructure Deployment

### Using Terraform
```bash
cd infra

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var-file="terraform.tfvars"

# Deploy infrastructure
terraform apply
```

### Using CloudFormation
```bash
# Deploy the CloudFormation stack
aws cloudformation create-stack \
  --stack-name predictive-maintenance \
  --template-body file://infra/cloudformation_template.yaml \
  --parameters file://infra/parameters.json \
  --capabilities CAPABILITY_IAM
```

## Verification and Testing

### 1. Verify Model Deployment
```bash
# List endpoints
python scripts/deploy_model.py --list-endpoints

# Test endpoint
python scripts/deploy_model.py --test-endpoint --endpoint-name your-endpoint-name
```

### 2. Test Data Pipeline
```python
# Test sensor data collection
python -c "
from src.iot.sensor_data_collector import SensorDataCollector
from src.utils.config_loader import load_config

config = load_config('config.yaml')
collector = SensorDataCollector(config)

# Generate test data
import time
collector.start_collection()
time.sleep(30)
collector.stop_collection()
print('Data collection test completed')
"
```

### 3. Verify Alerts
```python
# Test alerting system
python -c "
from src.monitoring.alerting_system import AlertingSystem, AlertType, AlertSeverity
from src.utils.config_loader import load_config

config = load_config('config.yaml')
alerting = AlertingSystem(config)
alerting.start_processing()

# Create test alert
alerting.create_alert(
    AlertType.EQUIPMENT_FAILURE,
    AlertSeverity.HIGH,
    'test_device',
    'Test alert from deployment verification'
)

import time
time.sleep(5)
alerting.stop_processing()
print('Alert test completed')
"
```

## Monitoring and Maintenance

### CloudWatch Dashboards
1. Navigate to AWS CloudWatch
2. Create custom dashboard for:
   - SageMaker endpoint metrics
   - IoT message rates
   - Lambda function performance
   - SNS notification delivery

### Log Analysis
```bash
# View application logs
tail -f logs/application.log

# View Docker logs
docker-compose logs -f predictive-maintenance

# View CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/
```

### Model Performance Monitoring
```python
# Monitor model performance
from src.deployment.endpoint_monitor import EndpointMonitor
from src.utils.config_loader import load_config

config = load_config('config.yaml')
monitor = EndpointMonitor(config)

# Get endpoint metrics
metrics = monitor.get_endpoint_metrics('your-endpoint-name')
print(f"Endpoint metrics: {metrics}")
```

## Troubleshooting

### Common Issues

#### 1. Model Training Fails
```bash
# Check training logs
cat logs/application.log | grep ERROR

# Verify data format
python -c "
import pandas as pd
data = pd.read_csv('data/raw/your_data.csv')
print(data.info())
print(data.head())
"
```

#### 2. SageMaker Deployment Fails
- Verify IAM role permissions
- Check S3 bucket access
- Ensure model artifacts are properly packaged

#### 3. IoT Data Not Flowing
- Verify AWS IoT certificates
- Check MQTT connectivity
- Validate message format

#### 4. Alerts Not Received
- Check SNS topic configuration
- Verify email subscriptions
- Test SNS publish permissions

### Performance Optimization

#### Model Optimization
```bash
# Retrain with more data
python scripts/train_model.py --config config.yaml --epochs 200

# Use different instance types
python scripts/deploy_model.py --instance-type ml.c5.xlarge
```

#### Cost Optimization
- Use Spot instances for training
- Implement auto-scaling for endpoints
- Archive old data to cheaper storage classes

## Security Best Practices

### IAM Permissions
- Use least-privilege access
- Rotate access keys regularly
- Enable MFA for sensitive operations

### Data Encryption
- Enable S3 bucket encryption
- Use HTTPS for all communications
- Encrypt sensitive configuration values

### Network Security
- Deploy in private VPC subnets
- Use security groups to restrict access
- Enable VPC Flow Logs for monitoring

## Backup and Recovery

### Data Backup
```bash
# Backup training data
aws s3 sync data/processed/ s3://your-backup-bucket/data/

# Backup model artifacts
aws s3 sync data/models/ s3://your-backup-bucket/models/
```

### Disaster Recovery
- Implement cross-region replication
- Maintain infrastructure as code
- Test recovery procedures regularly

## Scaling Considerations

### Horizontal Scaling
- Use multiple SageMaker endpoints
- Implement load balancing
- Scale IoT device connections

### Vertical Scaling
- Upgrade instance types
- Increase storage capacity
- Optimize model architecture

For additional support, refer to the [AWS documentation](https://docs.aws.amazon.com/) or create an issue in the project repository.
