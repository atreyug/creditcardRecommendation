from setuptools import setup, find_packages

setup(
    name="credit-card-env",
    version="1.0.0",
    description="Credit Card Recommendation System — OpenEnv Environment",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0.0,<3.0.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "openai>=1.30.0",
        "pandas>=2.1.0",
        "openpyxl>=3.1.0",
        "pyyaml>=6.0.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
