#!/usr/bin/env python3
"""
Training script for the predictive maintenance model.
Handles the complete training pipeline from data loading to model evaluation.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import yaml
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.model_trainer import ModelTrainer
from models.predictive_model import create_sample_data
from utils.config_loader import load_config
from utils.logger import setup_logger
from utils.aws_utils import AWSUtils


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train predictive maintenance model')

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--data',
        type=str,
        help='Path to training data file (optional - will generate synthetic data if not provided)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for trained model (overrides config)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size (overrides config)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without training'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def validate_config(config):
    """Validate configuration for required parameters."""
    required_sections = ['data', 'model', 'aws']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate data configuration
    data_config = config['data']
    required_data_params = ['features', 'target_column', 'preprocessing']

    for param in required_data_params:
        if param not in data_config:
            raise ValueError(f"Missing required data parameter: {param}")

    # Validate model configuration
    model_config = config['model']
    required_model_params = ['architecture', 'training']

    for param in required_model_params:
        if param not in model_config:
            raise ValueError(f"Missing required model parameter: {param}")

    # Validate AWS configuration
    aws_config = config['aws']
    required_aws_params = ['region', 'sagemaker', 's3']

    for param in required_aws_params:
        if param not in aws_config:
            raise ValueError(f"Missing required AWS parameter: {param}")

    print("✓ Configuration validation passed")


def prepare_training_data(data_path, config, logger):
    """Prepare training data from file or generate synthetic data."""
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading training data from {data_path}")

        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        # Save to processed data directory
        processed_path = os.path.join(config['data']['processed_data_path'], 'training_data.csv')
        os.makedirs(config['data']['processed_data_path'], exist_ok=True)
        data.to_csv(processed_path, index=False)

        return processed_path

    else:
        logger.info("Generating synthetic training data...")

        # Generate synthetic data
        n_samples = 50000  # Generate enough data for robust training
        synthetic_data = create_sample_data(n_samples=n_samples)

        # Convert to DataFrame
        feature_columns = config['data']['features'] + [config['data']['target_column']]
        data = pd.DataFrame(synthetic_data, columns=feature_columns)

        # Add timestamp and device_id columns
        timestamps = pd.date_range(
            start='2023-01-01', 
            periods=n_samples, 
            freq='10S'  # 10 second intervals
        )
        data['timestamp'] = timestamps
        data['device_id'] = np.random.choice([f'device_{i:03d}' for i in range(1, 11)], n_samples)

        # Save synthetic data
        processed_path = os.path.join(config['data']['processed_data_path'], 'synthetic_training_data.csv')
        os.makedirs(config['data']['processed_data_path'], exist_ok=True)
        data.to_csv(processed_path, index=False)

        logger.info(f"Generated {n_samples} synthetic samples and saved to {processed_path}")
        return processed_path


def main():
    """Main training function."""
    args = parse_arguments()

    try:
        # Load configuration
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Override configuration with command line arguments
        if args.output_dir:
            config['data']['model_path'] = args.output_dir

        if args.epochs:
            config['model']['training']['epochs'] = args.epochs

        if args.batch_size:
            config['model']['training']['batch_size'] = args.batch_size

        # Set up logging
        if args.verbose:
            config['logging'] = config.get('logging', {})
            config['logging']['level'] = 'DEBUG'

        logger = setup_logger(__name__, config.get('logging', {}))

        # Validate configuration
        validate_config(config)

        if args.validate_only:
            print("✓ Configuration validation completed successfully")
            return 0

        # Set up AWS resources
        logger.info("Setting up AWS resources...")
        aws_utils = AWSUtils(config)
        aws_resources = aws_utils.setup_aws_resources()
        logger.info(f"AWS resources: {aws_resources}")

        # Prepare training data
        data_path = prepare_training_data(args.data, config, logger)

        # Initialize model trainer
        logger.info("Initializing model trainer...")
        trainer = ModelTrainer(config)

        # Run complete training pipeline
        logger.info("Starting training pipeline...")
        start_time = datetime.now()

        results = trainer.run_complete_training_pipeline(data_path)

        end_time = datetime.now()
        training_duration = end_time - start_time

        # Log results
        logger.info("Training completed successfully!")
        logger.info(f"Training duration: {training_duration}")
        logger.info(f"Final validation accuracy: {results['evaluation_results']['metrics']['accuracy']:.4f}")
        logger.info(f"Final validation precision: {results['evaluation_results']['metrics']['precision']:.4f}")
        logger.info(f"Final validation recall: {results['evaluation_results']['metrics']['recall']:.4f}")

        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Training Duration: {training_duration}")
        print(f"Total Samples: {results['data_info']['total_samples']:,}")
        print(f"Training Samples: {results['data_info']['training_samples']:,}")
        print(f"Validation Samples: {results['data_info']['validation_samples']:,}")
        print(f"Feature Count: {results['data_info']['feature_count']}")
        print(f"Sequence Length: {results['data_info']['sequence_length']}")
        print("\nFinal Metrics:")
        print(f"  Accuracy:  {results['evaluation_results']['metrics']['accuracy']:.4f}")
        print(f"  Precision: {results['evaluation_results']['metrics']['precision']:.4f}")
        print(f"  Recall:    {results['evaluation_results']['metrics']['recall']:.4f}")
        print(f"  F1-Score:  {results['evaluation_results']['metrics']['f1_score']:.4f}")
        print(f"  ROC-AUC:   {results['evaluation_results']['metrics']['roc_auc']:.4f}")
        print("="*60)

        # Save training summary
        summary_path = os.path.join(config['data']['model_path'], 'training_summary.yaml')
        with open(summary_path, 'w') as f:
            yaml.dump({
                'training_completed': end_time.isoformat(),
                'training_duration_seconds': training_duration.total_seconds(),
                'results': results
            }, f, default_flow_style=False)

        logger.info(f"Training summary saved to {summary_path}")

        return 0

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1

    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
