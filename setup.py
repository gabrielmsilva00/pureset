from setuptools import setup, find_packages
import src.pureset as pkg

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=pkg.__title__,
    version=pkg.__version__,
    author=pkg.__author__,
    author_email=pkg.__contact__,
    description=pkg.__desc__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=pkg.__repo__,
    project_urls={
        "Bug Tracker": pkg.__repo__ + "/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
)