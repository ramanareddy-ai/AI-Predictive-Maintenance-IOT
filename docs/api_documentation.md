# API Documentation

## REST API Endpoints

The predictive maintenance system provides RESTful API endpoints for integration with external systems.

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
All API requests require authentication using JWT tokens.

```bash
# Get authentication token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

### Endpoints

#### 1. Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-19T23:16:00Z",
  "version": "1.0.0"
}
```

#### 2. Predict Equipment Failure
```
POST /predict
```

**Request Body:**
```json
{
  "device_id": "device_001",
  "sensor_data": {
    "temperature": 75.5,
    "vibration": 2.1,
    "pressure": 45.2,
    "rotation_speed": 3500,
    "torque": 78.5,
    "tool_wear": 15.2,
    "power_consumption": 102.3,
    "operating_hours": 150.5
  }
}
```

**Response:**
```json
{
  "device_id": "device_001",
  "failure_probability": 0.12,
  "risk_level": "low",
  "recommended_action": "continue_monitoring",
  "next_maintenance_window": "2025-09-15T10:00:00Z",
  "timestamp": "2025-08-19T23:16:00Z"
}
```

#### 3. Get Device Status
```
GET /devices/{device_id}/status
```

**Response:**
```json
{
  "device_id": "device_001",
  "status": "operational",
  "last_reading": "2025-08-19T23:15:00Z",
  "current_metrics": {
    "temperature": 72.1,
    "vibration": 1.8,
    "pressure": 46.5
  },
  "alerts": [
    {
      "alert_id": "alert_123",
      "severity": "medium",
      "message": "Vibration levels slightly elevated",
      "timestamp": "2025-08-19T22:30:00Z"
    }
  ]
}
```

#### 4. Submit Sensor Data
```
POST /data/sensors
```

**Request Body:**
```json
{
  "device_id": "device_001",
  "timestamp": "2025-08-19T23:16:00Z",
  "readings": {
    "temperature": 72.1,
    "vibration": 1.8,
    "pressure": 46.5,
    "rotation_speed": 3480,
    "torque": 76.2,
    "tool_wear": 15.8,
    "power_consumption": 98.7,
    "operating_hours": 151.2
  }
}
```

**Response:**
```json
{
  "status": "accepted",
  "message": "Sensor data received and processed",
  "timestamp": "2025-08-19T23:16:00Z"
}
```

#### 5. Get Alerts
```
GET /alerts?status=active&severity=high
```

**Response:**
```json
{
  "alerts": [
    {
      "alert_id": "alert_456",
      "device_id": "device_002",
      "alert_type": "equipment_failure",
      "severity": "high",
      "message": "High failure probability detected: 85%",
      "timestamp": "2025-08-19T23:10:00Z",
      "acknowledged": false,
      "recommended_action": "Schedule immediate maintenance"
    }
  ],
  "total_count": 1,
  "page": 1,
  "limit": 50
}
```

#### 6. Acknowledge Alert
```
PUT /alerts/{alert_id}/acknowledge
```

**Request Body:**
```json
{
  "user": "maintenance_tech_01",
  "notes": "Maintenance scheduled for tomorrow"
}
```

**Response:**
```json
{
  "status": "acknowledged",
  "alert_id": "alert_456",
  "acknowledged_by": "maintenance_tech_01",
  "acknowledged_at": "2025-08-19T23:16:00Z"
}
```

## Error Responses

### 400 Bad Request
```json
{
  "error": "bad_request",
  "message": "Invalid sensor data format",
  "details": {
    "field": "temperature",
    "issue": "Value out of valid range"
  }
}
```

### 401 Unauthorized
```json
{
  "error": "unauthorized",
  "message": "Invalid or expired authentication token"
}
```

### 404 Not Found
```json
{
  "error": "not_found",
  "message": "Device not found",
  "device_id": "device_999"
}
```

### 500 Internal Server Error
```json
{
  "error": "internal_server_error",
  "message": "An unexpected error occurred",
  "request_id": "req_12345"
}
```

## Rate Limiting

API requests are limited to 100 requests per minute per client. When the limit is exceeded:

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests",
  "retry_after": 60
}
```

## WebSocket API

For real-time updates, the system provides WebSocket connections:

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Subscribe to device updates
ws.send(JSON.stringify({
  "action": "subscribe",
  "topic": "device_updates",
  "device_id": "device_001"
}));

// Receive real-time updates
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Device update:', data);
};
```

## SDK Examples

### Python SDK
```python
from predictive_maintenance_client import PredictiveMaintenanceClient

client = PredictiveMaintenanceClient(
    base_url="http://localhost:8000",
    api_key="your_api_key"
)

# Make prediction
result = client.predict_failure(
    device_id="device_001",
    sensor_data={
        "temperature": 75.5,
        "vibration": 2.1,
        "pressure": 45.2
    }
)

print(f"Failure probability: {result.failure_probability}")
```

### JavaScript SDK
```javascript
import { PredictiveMaintenanceClient } from 'predictive-maintenance-js';

const client = new PredictiveMaintenanceClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your_api_key'
});

// Make prediction
const result = await client.predictFailure({
  deviceId: 'device_001',
  sensorData: {
    temperature: 75.5,
    vibration: 2.1,
    pressure: 45.2
  }
});

console.log(`Failure probability: ${result.failureProbability}`);
```

For more information, see the [full API specification](./openapi.yaml).
