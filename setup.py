from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="trust-network",
    version="1.0.0",
    author="Trust Network Contributors",
    author_email="contact@trustnetwork.org",
    description="A comprehensive trust scoring framework for decentralized token networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trustnetwork/trust-network",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "viz": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "trust-network=trust_network.cli:main",
        ],
    },
)