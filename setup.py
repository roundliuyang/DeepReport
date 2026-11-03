"""
DeepReport setup configuration
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="deepreport",
    version="1.0.0",
    author="DeepReport Team",
    author_email="contact@deepreport.ai",
    description="AI-powered financial research and report generation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/DeepReport",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Markup :: HTML",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "all": read_requirements("requirements.txt"),
    },
    entry_points={
        "console_scripts": [
            "deepreport=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["templates/*.html", "static/*", "docs/*"],
    },
    keywords="ai, financial, research, reports, agents, automation",
    project_urls={
        "Bug Reports": "https://github.com/your-username/DeepReport/issues",
        "Source": "https://github.com/your-username/DeepReport",
        "Documentation": "https://deepreport.readthedocs.io/",
    },
)