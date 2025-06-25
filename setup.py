import os
from glob import glob
from setuptools import setup

package_name = 'openpi_client'

setup(
    version='0.1.0',
    name=package_name,
    description='Starts a client for that publishes observations to the pi0 server.',
    packages=[package_name],
    package_dir={'': 'src'} ,
    entry_points={
        'console_scripts': [
            'pi0 = pi0:main'
        ],
    },
    install_requires=['websockets', 'pillow>=9.0.0'],
    maintainer='hashimismail',
    maintainer_email='Hashim.Ismail@cs.lth.se',
    license='TODO: License declaration',
)