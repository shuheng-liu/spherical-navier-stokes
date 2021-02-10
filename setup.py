import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="spherical-navier-stokes",  # Replace with your username
    version="0.1.0",
    author="Shuheng Liu",
    author_email="wish1104@icloud.com",
    description="Spherical Navier Stokes",
    url="https://github.com/shuheng-liu/spherical-navier-stokes",
    packages=setuptools.find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    setup_requires=['setuptools_scm'],
    include_package_data=True,
)
