import re
import sys

# Make sure that kornia is running on Python 3.6.0 or later
# (to avoid running into this bug: https://bugs.python.org/issue29246)

if sys.version_info < (3, 6, 0):
    raise RuntimeError("PyPose requires Python 3.6.0 or later.")


from setuptools import setup, find_packages

def find_version(file_path: str) -> str:
    version_file = open(file_path).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if not version_match:
        raise RuntimeError(f"Unable to find version string in {file_path}")
    return version_match.group(1)

VERSION = find_version("pypose/_version.py")


# NOTE: PyPose MUST only require PyTorch
requirements = ['torch==1.13.1', 'packaging']

# open readme file and set long description
with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()


def load_requirements(filename: str):
    with open(filename) as f:
        return [x.strip() for x in f.readlines() if "-r" != x[0:2]]


requirements_extras = {"main":   load_requirements("requirements/main.txt"), 
                       "dev": load_requirements("requirements/dev.txt")}

requirements_extras["all"] = requirements_extras["main"] + requirements_extras["dev"]


if __name__ == '__main__':
    setup(
        name = 'pypose',
        author = 'Chen Wang',
        version = VERSION,
        author_email = 'chenwang@dr.com',
        url = 'https://pypose.org',
        download_url = 'https://github.com/pypose/pypose',
        license = 'BSD 4-Clause License',
        description = 'To connect classic robotics with modern learning methods.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        python_requires='>=3.6',
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        packages=find_packages(exclude=('docs', 'test', 'examples')),
        data_files=[('', ['requirements/main.txt', 'requirements/dev.txt'])],
        zip_safe=True,
        install_requires = requirements,
        extras_require = requirements_extras,
        keywords=['robotics', 'deep learning', 'pytorch'],
        project_urls={
            "Bug Tracker": "https://github.com/pypose/pypose/issues",
            "Documentation": "https://pypose.org/docs",
            "Source Code": "https://github.com/pypose/pypose",
        },
        classifiers=[
            'Environment :: GPU',
            'Environment :: Console',
            'Natural Language :: English',
            # How mature is this project? Common values are
            #   3 - Alpha, 4 - Beta, 5 - Production/Stable
            'Development Status :: 3 - Alpha',
            # Indicate who your project is intended for
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Topic :: Software Development :: Libraries',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            # Pick your license as you wish
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
    )
