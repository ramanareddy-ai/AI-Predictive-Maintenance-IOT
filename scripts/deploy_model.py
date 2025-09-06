#!/usr/bin/env python3
"""
Deployment script for the predictive maintenance model.
Handles model packaging and deployment to AWS SageMaker.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import yaml
import glob

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from deployment.sagemaker_deploy import SageMakerDeployment, deploy_model_to_sagemaker
from utils.config_loader import load_config
from utils.logger import setup_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deploy predictive maintenance model to AWS SageMaker')

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained model file (.h5 or directory)'
    )

    parser.add_argument(
        '--endpoint-name',
        type=str,
        help='Custom endpoint name (optional)'
    )

    parser.add_argument(
        '--instance-type',
        type=str,
        help='SageMaker instance type for endpoint (overrides config)'
    )

    parser.add_argument(
        '--instance-count',
        type=int,
        help='Number of instances for endpoint (overrides config)'
    )

    parser.add_argument(
        '--auto-scaling',
        action='store_true',
        help='Enable auto-scaling for the endpoint'
    )

    parser.add_argument(
        '--test-endpoint',
        action='store_true',
        help='Test the deployed endpoint with sample data'
    )

    parser.add_argument(
        '--delete-endpoint',
        type=str,
        help='Delete an existing endpoint by name'
    )

    parser.add_argument(
        '--list-endpoints',
        action='store_true',
        help='List all existing endpoints'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def find_latest_model(model_dir):
    """Find the latest trained model in the model directory."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Look for .h5 model files
    h5_files = glob.glob(os.path.join(model_dir, "*.h5"))

    if not h5_files:
        raise FileNotFoundError(f"No .h5 model files found in {model_dir}")

    # Return the most recently modified file
    latest_model = max(h5_files, key=os.path.getmtime)
    return latest_model


def test_endpoint(deployment, endpoint_name, config, logger):
    """Test the deployed endpoint with sample data."""
    try:
        import numpy as np
        from models.predictive_model import create_sample_data

        logger.info("Testing endpoint with sample data...")

        # Generate sample data
        sample_data = create_sample_data(n_samples=100)

        # Create sequences (same as training)
        sequence_length = config['data']['preprocessing']['sequence_length']
        n_features = len(config['data']['features'])

        # Take a single sequence for testing
        test_sequence = sample_data[:sequence_length, :n_features]
        test_sequence = test_sequence.reshape(1, sequence_length, n_features)

        # Make prediction
        result = deployment.predict(test_sequence)

        logger.info(f"Endpoint test successful!")
        logger.info(f"Sample prediction result: {result}")

        return True

    except Exception as e:
        logger.error(f"Endpoint test failed: {str(e)}")
        return False


def main():
    """Main deployment function."""
    args = parse_arguments()

    try:
        # Load configuration
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Override configuration with command line arguments
        if args.instance_type:
            config['aws']['sagemaker']['endpoint']['instance_type'] = args.instance_type

        if args.instance_count:
            config['aws']['sagemaker']['endpoint']['initial_instance_count'] = args.instance_count

        if args.auto_scaling:
            config['aws']['sagemaker']['endpoint']['auto_scaling']['enabled'] = True

        # Set up logging
        if args.verbose:
            config['logging'] = config.get('logging', {})
            config['logging']['level'] = 'DEBUG'

        logger = setup_logger(__name__, config.get('logging', {}))

        # Initialize deployment handler
        deployment = SageMakerDeployment(config)

        # Handle different operations
        if args.delete_endpoint:
            logger.info(f"Deleting endpoint: {args.delete_endpoint}")
            deployment.delete_endpoint(args.delete_endpoint)
            print(f"✓ Endpoint deletion initiated: {args.delete_endpoint}")
            return 0

        if args.list_endpoints:
            logger.info("Listing all endpoints...")
            endpoints = deployment.list_endpoints()

            if endpoints:
                print("\nExisting SageMaker Endpoints:")
                print("-" * 60)
                for endpoint in endpoints:
                    print(f"Name: {endpoint['name']}")
                    print(f"Status: {endpoint['status']}")
                    print(f"Created: {endpoint['creation_time']}")
                    print(f"Modified: {endpoint['last_modified_time']}")
                    print("-" * 60)
            else:
                print("No endpoints found")

            return 0

        # Deploy model
        if args.model_path:
            model_path = args.model_path
        else:
            # Find latest model
            model_dir = config['data']['model_path']
            model_path = find_latest_model(model_dir)
            logger.info(f"Using latest model: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"\nDeploying model: {model_path}")
        print(f"Endpoint name: {args.endpoint_name or 'Auto-generated'}")
        print(f"Instance type: {config['aws']['sagemaker']['endpoint']['instance_type']}")
        print(f"Instance count: {config['aws']['sagemaker']['endpoint']['initial_instance_count']}")

        # Confirm deployment
        response = input("\nProceed with deployment? (y/N): ")
        if response.lower() != 'y':
            print("Deployment cancelled")
            return 0

        # Deploy model
        logger.info("Starting model deployment...")
        start_time = datetime.now()

        endpoint_name, deployment = deploy_model_to_sagemaker(
            config=config,
            model_path=model_path,
            endpoint_name=args.endpoint_name
        )

        end_time = datetime.now()
        deployment_duration = end_time - start_time

        # Test endpoint if requested
        if args.test_endpoint:
            test_success = test_endpoint(deployment, endpoint_name, config, logger)
            if not test_success:
                logger.warning("Endpoint test failed, but deployment was successful")

        # Print deployment summary
        print("\n" + "="*60)
        print("DEPLOYMENT SUMMARY")
        print("="*60)
        print(f"Endpoint Name: {endpoint_name}")
        print(f"Model Path: {model_path}")
        print(f"Deployment Duration: {deployment_duration}")
        print(f"Instance Type: {config['aws']['sagemaker']['endpoint']['instance_type']}")
        print(f"Instance Count: {config['aws']['sagemaker']['endpoint']['initial_instance_count']}")
        print(f"Auto-scaling: {'Enabled' if config['aws']['sagemaker']['endpoint']['auto_scaling']['enabled'] else 'Disabled'}")
        print(f"Region: {config['aws']['region']}")
        print("\n✓ Model deployed successfully!")
        print("="*60)

        # Save deployment info
        deployment_info = {
            'endpoint_name': endpoint_name,
            'model_path': model_path,
            'deployment_time': end_time.isoformat(),
            'deployment_duration_seconds': deployment_duration.total_seconds(),
            'config': {
                'instance_type': config['aws']['sagemaker']['endpoint']['instance_type'],
                'instance_count': config['aws']['sagemaker']['endpoint']['initial_instance_count'],
                'auto_scaling_enabled': config['aws']['sagemaker']['endpoint']['auto_scaling']['enabled']
            }
        }

        deployment_info_path = os.path.join(config['data']['model_path'], 'deployment_info.yaml')
        with open(deployment_info_path, 'w') as f:
            yaml.dump(deployment_info, f, default_flow_style=False)

        logger.info(f"Deployment info saved to {deployment_info_path}")

        return 0

    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
        return 1

    except Exception as e:
        print(f"\nDeployment failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
