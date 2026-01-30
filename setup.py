"""Setup script for Magpie AI SDK."""
from setuptools import setup, find_packages

setup(
    name="magpie-ai",
    version="0.2.8",
    author="Magpie Team",
    author_email="team@magpie.dev",
    description="Enterprise-grade LLM middleware for monitoring and metadata tracking",
    long_description="Enterprise-grade LLM middleware SDK for monitoring and metadata tracking",
    long_description_content_type="text/plain",
    url="https://github.com/magpie-ai/sdk",
    packages=find_packages(),
    package_data={
        "magpie_ai": ["py.typed"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "httpx>=0.26.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ]
    }
)
