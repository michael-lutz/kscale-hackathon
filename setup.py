from setuptools import setup, find_packages  # type: ignore

setup(
    name="ube",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "attrs",
        # Add other dependencies here
    ],
    author="Michael Lutz",
    author_email="michaeljeffreylutz@gmail.com",
    description="KScale Hackathon Implementation of UBE",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/michael-lutz/kscale-hackathon",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
