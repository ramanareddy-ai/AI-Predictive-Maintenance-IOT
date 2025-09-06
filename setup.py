from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-predictive-maintenance-iot",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@company.com",
    description="AI-Driven Predictive Maintenance for IoT Devices using TensorFlow and AWS SageMaker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ai-predictive-maintenance-iot",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/ai-predictive-maintenance-iot/issues",
        "Documentation": "https://github.com/your-username/ai-predictive-maintenance-iot/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.1.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "ipywidgets>=8.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "train-model=scripts.train_model:main",
            "deploy-model=scripts.deploy_model:main",
            "run-pipeline=scripts.data_pipeline:main",
            "setup-aws=scripts.setup_aws_resources:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
