"""
Setup configuration for AI Stock Portfolio Optimizer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-stock-portfolio-optimizer",
    version="1.0.0",
    author="Mekeda Bogdan Sergeevich",
    author_email="titrityt73250@gmail.com",
    description="AI-powered stock portfolio optimization system with ML predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/T1Trit/ai-stock-portfolio-optimizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "yfinance>=0.2.28",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.1",
        "tensorflow>=2.13.0",
        "plotly>=5.15.0",
        "streamlit>=1.25.0",
    ],
)