#!/usr/bin/env python3
"""
ReGenNexus Core - Setup Script

This script installs the ReGenNexus Core Universal Agent Protocol.
"""

from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README for long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="regennexus-core",
    version="0.1.1",
    description="ReGenNexus Core - Universal Agent Protocol",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ReGen Designs LLC",
    author_email="info@regendesigns.com",
    url="https://github.com/ReGenNow/ReGenNexus",
    packages=find_packages(where="."),  # Look for packages in the root directory
    package_dir={"": "."},  # Map the root package to the current directory
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "raspberry_pi": [
            "RPi.GPIO>=0.7.0",
            "picamera>=1.13"
        ],
        "jetson": [
            "jetson-stats>=3.1.0",
            "Jetson.GPIO>=2.0.17"
        ],
        "ros": [
            "rclpy>=1.0.0"
        ],
        "azure": [
            "azure-iot-device>=2.12.0"
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.3.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "regennexus-registry=regennexus.registry.cli:main",
            "regennexus-client=regennexus.protocol.cli:main",
            "regennexus-ros-amber-b1=examples.ros_integration.amber_b1.amber_b1_arm_example:main",
            "regennexus-ros-amber-b1-moveit=examples.ros_integration.amber_b1.amber_b1_moveit_example:main",
            "regennexus-ros-amber-lucid1-moveit=examples.ros_integration.amber_lucid1.amber_lucid1_moveit_example:main",
            "regennexus-ros-amber-lucid1=examples.ros_integration.amber_lucid1.amber_lucid1_arm_example:main",
        ],
    },
)