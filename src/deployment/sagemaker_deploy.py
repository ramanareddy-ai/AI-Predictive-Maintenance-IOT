"""
AWS SageMaker Deployment Module for Predictive Maintenance
Handles model deployment, endpoint creation, and inference on AWS SageMaker.
"""

import os
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
import tarfile
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np


class SageMakerDeployment:
    """
    AWS SageMaker deployment handler for predictive maintenance models.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the SageMaker deployment handler."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # AWS configuration
        self.aws_region = config['aws']['region']
        self.sagemaker_role = config['aws']['sagemaker']['role_arn']
        self.s3_bucket = config['aws']['s3']['bucket']
        self.model_prefix = config['aws']['s3']['model_prefix']

        # Initialize AWS clients
        self.sagemaker_session = sagemaker.Session()
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.aws_region)
        self.s3_client = boto3.client('s3', region_name=self.aws_region)

        # Deployment configuration
        self.instance_type = config['aws']['sagemaker']['instance_type']
        self.endpoint_instance_type = config['aws']['sagemaker']['endpoint']['instance_type']
        self.initial_instance_count = config['aws']['sagemaker']['endpoint']['initial_instance_count']

        self.predictor = None

    def create_model_package(self, model_path: str) -> str:
        """Create a model package for SageMaker deployment."""
        self.logger.info("Creating model package for SageMaker deployment...")

        # Create temporary directory for packaging
        package_dir = f"/tmp/sagemaker_model_{int(time.time())}"
        os.makedirs(package_dir, exist_ok=True)

        try:
            # Copy model files
            self._prepare_model_files(model_path, package_dir)

            # Create inference script
            self._create_inference_script(package_dir)

            # Create model.tar.gz
            model_tar_path = f"{package_dir}/model.tar.gz"
            self._create_model_tarball(package_dir, model_tar_path)

            # Upload to S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"{self.model_prefix}predictive_maintenance_model_{timestamp}.tar.gz"
            s3_uri = f"s3://{self.s3_bucket}/{s3_key}"

            self.s3_client.upload_file(model_tar_path, self.s3_bucket, s3_key)
            self.logger.info(f"Model package uploaded to {s3_uri}")

            return s3_uri

        except Exception as e:
            self.logger.error(f"Failed to create model package: {str(e)}")
            raise
        finally:
            # Cleanup temporary files
            import shutil
            if os.path.exists(package_dir):
                shutil.rmtree(package_dir)

    def _prepare_model_files(self, model_path: str, package_dir: str) -> None:
        """Prepare model files for packaging."""
        # Copy model file
        if os.path.isfile(model_path):
            import shutil
            model_filename = os.path.basename(model_path)
            dest_path = os.path.join(package_dir, model_filename)
            shutil.copy2(model_path, dest_path)

        # Copy scaler if exists
        scaler_path = os.path.join(os.path.dirname(model_path), "scaler.joblib")
        if os.path.exists(scaler_path):
            import shutil
            shutil.copy2(scaler_path, os.path.join(package_dir, "scaler.joblib"))

    def _create_inference_script(self, package_dir: str) -> None:
        """Create the inference script for SageMaker."""
        inference_lines = [
            "import os",
            "import json", 
            "import numpy as np",
            "import tensorflow as tf",
            "from tensorflow import keras",
            "import joblib",
            "import logging",
            "",
            "logging.basicConfig(level=logging.INFO)",
            "logger = logging.getLogger(__name__)",
            "",
            "def model_fn(model_dir):",
            "    '''Load the model for inference'''",
            "    try:",
            "        logger.info(f'Loading model from {model_dir}')",
            "        ",
            "        # Load TensorFlow model",
            "        model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]",
            "        if model_files:",
            "            model_path = os.path.join(model_dir, model_files[0])",
            "            model = keras.models.load_model(model_path)",
            "            logger.info('TensorFlow model loaded successfully')",
            "        else:",
            "            model = keras.models.load_model(os.path.join(model_dir, 'model'))",
            "            logger.info('SavedModel loaded successfully')",
            "        ",
            "        # Load scaler",
            "        scaler_path = os.path.join(model_dir, 'scaler.joblib')",
            "        scaler = None",
            "        if os.path.exists(scaler_path):",
            "            scaler = joblib.load(scaler_path)",
            "            logger.info('Scaler loaded successfully')",
            "        ",
            "        return {'model': model, 'scaler': scaler}",
            "        ",
            "    except Exception as e:",
            "        logger.error(f'Error loading model: {str(e)}')",
            "        raise",
            "",
            "def input_fn(request_body, request_content_type):",
            "    '''Parse input data for inference'''",
            "    if request_content_type == 'application/json':",
            "        input_data = json.loads(request_body)",
            "        return np.array(input_data['instances'])",
            "    else:",
            "        raise ValueError(f'Unsupported content type: {request_content_type}')",
            "",
            "def predict_fn(input_data, model_artifacts):",
            "    '''Make predictions using the loaded model'''",
            "    try:",
            "        model = model_artifacts['model']",
            "        scaler = model_artifacts['scaler']",
            "        ",
            "        # Preprocess input data",
            "        if scaler is not None:",
            "            original_shape = input_data.shape",
            "            if len(original_shape) == 3:",
            "                input_data = input_data.reshape(-1, original_shape[-1])",
            "                input_data = scaler.transform(input_data)",
            "                input_data = input_data.reshape(original_shape)",
            "            else:",
            "                input_data = scaler.transform(input_data)",
            "        ",
            "        # Make prediction",
            "        predictions = model.predict(input_data)",
            "        return predictions.tolist()",
            "        ",
            "    except Exception as e:",
            "        logger.error(f'Error making prediction: {str(e)}')",
            "        raise",
            "",
            "def output_fn(prediction, content_type):",
            "    '''Format the prediction output'''",
            "    if content_type == 'application/json':",
            "        return json.dumps({",
            "            'predictions': prediction,",
            "            'model_name': 'predictive_maintenance',",
            "            'version': '1.0'",
            "        })",
            "    else:",
            "        raise ValueError(f'Unsupported content type: {content_type}')"
        ]

        with open(os.path.join(package_dir, "inference.py"), "w") as f:
            f.write("\n".join(inference_lines))

    def _create_model_tarball(self, package_dir: str, tar_path: str) -> None:
        """Create a compressed tarball of the model package."""
        with tarfile.open(tar_path, "w:gz") as tar:
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    if file != os.path.basename(tar_path):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, package_dir)
                        tar.add(file_path, arcname=arcname)

        self.logger.info(f"Model tarball created at {tar_path}")

    def create_model(self, model_s3_uri: str, model_name: str = None) -> str:
        """Create a SageMaker model from the S3 model package."""
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            model_name = f"predictive-maintenance-model-{timestamp}"

        self.logger.info(f"Creating SageMaker model: {model_name}")

        # Create TensorFlow model
        tensorflow_model = TensorFlowModel(
            model_data=model_s3_uri,
            role=self.sagemaker_role,
            framework_version='2.13',
            py_version='py310',
            entry_point='inference.py',
            name=model_name,
            sagemaker_session=self.sagemaker_session
        )

        self.tensorflow_model = tensorflow_model
        self.logger.info(f"SageMaker model created: {model_name}")
        return model_name

    def deploy_endpoint(self, model_name: str = None, endpoint_name: str = None) -> str:
        """Deploy the model to a SageMaker endpoint."""
        if endpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            endpoint_name = f"predictive-maintenance-endpoint-{timestamp}"

        self.logger.info(f"Deploying model to endpoint: {endpoint_name}")

        try:
            # Deploy the model
            from sagemaker.serializers import JSONSerializer
            from sagemaker.deserializers import JSONDeserializer

            self.predictor = self.tensorflow_model.deploy(
                initial_instance_count=self.initial_instance_count,
                instance_type=self.endpoint_instance_type,
                endpoint_name=endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )

            self.endpoint_name = endpoint_name
            self.logger.info(f"Model successfully deployed to endpoint: {endpoint_name}")

            return endpoint_name

        except Exception as e:
            self.logger.error(f"Failed to deploy endpoint: {str(e)}")
            raise

    def predict(self, data: np.ndarray) -> Dict[str, Any]:
        """Make predictions using the deployed endpoint."""
        if self.predictor is None:
            raise ValueError("No endpoint deployed. Call deploy_endpoint first.")

        try:
            # Prepare input data
            input_data = {
                'instances': data.tolist() if isinstance(data, np.ndarray) else data
            }

            # Make prediction
            result = self.predictor.predict(input_data)
            return result

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def delete_endpoint(self, endpoint_name: str = None) -> None:
        """Delete a SageMaker endpoint."""
        if endpoint_name is None:
            endpoint_name = getattr(self, 'endpoint_name', None)

        if endpoint_name is None:
            self.logger.warning("No endpoint name provided")
            return

        try:
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            self.logger.info(f"Endpoint {endpoint_name} deletion initiated")

        except Exception as e:
            self.logger.error(f"Failed to delete endpoint: {str(e)}")


def deploy_model_to_sagemaker(config: Dict[str, Any], model_path: str, 
                            endpoint_name: str = None) -> Tuple[str, SageMakerDeployment]:
    """Complete model deployment pipeline to SageMaker."""
    deployment = SageMakerDeployment(config)

    # Create model package
    model_s3_uri = deployment.create_model_package(model_path)

    # Create SageMaker model
    model_name = deployment.create_model(model_s3_uri)

    # Deploy endpoint
    endpoint_name = deployment.deploy_endpoint(model_name, endpoint_name)

    return endpoint_name, deployment
