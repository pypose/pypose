# For developers

## 1. Contributing to Documentation

## 1.1 Build docs locally

1. Sphinx docs come with a makefile build system. To preview, build PyPose locally and

```bash
cd docs
pip install -r requirements.txt
make html
```

2. Then open the generated HTML page: `docs/build/html/index.html`.

3. To clean and rebuild the doc:
```
make clean
```


## 1.2 Writing documentation

1. For the most simple case, you only need to edit the Python files and add docstring to functions following [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

2. Sometimes you may need to edit rst files like [lietensor.rst](docs/source/lietensor.rst), e.g., adding a new doc page.
More details can be found at [rst markdown](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

3. Create a pull request.

## 2. Contributing to New Features

Create a pull request. More details to be added.
