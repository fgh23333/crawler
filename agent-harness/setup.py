"""Setup script for cli-anything-crawler."""

from setuptools import setup, find_namespace_packages

setup(
    name="cli-anything-crawler",
    version="1.0.0",
    description="CLI harness for Political Theory Question Bank Crawler and Knowledge Graph System",
    packages=find_namespace_packages(include=["cli_anything.*"]),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[
        "click>=7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cli-anything-crawler=cli_anything.crawler.crawler_cli:main",
        ],
    },
    author="CLI-Anything",
    license="MIT",
)
