# Table of Contents

- [Writing documentation](#writing-documentation)

## Writing documentation

You may follow the [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) when writing docstrings for Python script files. New doc pages written in [reStructuredText markdown](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) can be added under [`docs/source`](docs/source). Make sure to refer to the newly added page in the body or `toctree` of an existing page.

### Build docs locally

Sphinx docs come with a makefile build system. To preview, build PyPose locally and

```bash
cd docs
pip install -r requirements.txt
make html
```

The HTML pages are then built to `docs/build/html`.
