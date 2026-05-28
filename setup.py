"""Installation script for the 'ur5_lab' python package."""

import os
import toml
from setuptools import setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

setup(
    name="ur5_lab",
    packages=["ur5_lab", "ur5_lab.tasks", "ur5_lab.assets"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=["psutil"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)
