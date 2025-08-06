#!/usr/bin/env python3
"""
setup.py
Video Monitoring System Package Setup

Package configuration for installation and distribution.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Video Monitoring System - Flexible AI-powered video monitoring"

# Read requirements
def read_requirements():
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

setup(
    name="video-monitoring-system",
    version="1.0.0",
    author="Video Monitoring Team",
    author_email="team@videomonitoring.com",
    description="Flexible AI-powered video monitoring system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/video-monitoring-system",
    
    # Package configuration
    packages=find_packages(),
    package_dir={'': '.'},
    
    # Include package data
    package_data={
        '': ['*.sql', '*.yml', '*.yaml', '*.json', '*.env.example'],
        'src': ['*.py'],
        'src.detection_scripts': ['*.py'],
    },
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Extra dependencies for different use cases
    extras_require={
        'gpu': ['torch[gpu]', 'torchvision[gpu]'],
        'cloud': ['google-cloud-storage', 'google-cloud-sql'],
        'dev': ['pytest', 'pytest-asyncio', 'black', 'flake8'],
        'docs': ['sphinx', 'sphinx-rtd-theme'],
        'monitoring': ['prometheus-client', 'grafana-api'],
    },
    
    # Python version requirement
    python_requires='>=3.9',
    
    # Entry points for command line scripts
    entry_points={
        'console_scripts': [
            'video-monitor=src.main:main',
            'vm-health=src.main:health_check',
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for package discovery
    keywords="video monitoring ai detection surveillance security computer-vision",
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/your-org/video-monitoring-system/issues',
        'Source': 'https://github.com/your-org/video-monitoring-system',
        'Documentation': 'https://video-monitoring-system.readthedocs.io/',
    },
    
    # Include additional files
    include_package_data=True,
    zip_safe=False,
)