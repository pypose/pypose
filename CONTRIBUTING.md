# Contributing to PyPose
Thank you for your interest in contributing to PyPose!
Our focus is on creating a state-of-the-art library for robotics research, and we are open to accepting various types of contributions.
Before you start coding for PyPose, it is important to inform the PyPose team of your intention to contribute and specify the type of contribution you plan to make.
These contributions may include but are not limited to:

1. If you notice a typographical error in PyPose's code or documentation, you can submit a Pull Request directly to fix it.
    - If the changes you wish to make to PyPose involve significant modifications, it is recommended that you create an issue first. Within the issue, describe the error and provide information on how to reproduce the bug. Other developers will then review the issue and engage in discussion.
    - If you require additional information regarding a specific issue, please do not hesitate to request it, and we will gladly provide it to you.
2. If you have an idea for a new feature in PyPose and wish to implement it, we recommend that you create an issue to post about your proposed feature. From there, we can discuss the design and implementation. Once we agree that the plan looks good, go ahead and implement it.

Once you implement and test your feature or bug fix, please submit a Pull Request.

## Workflow
This document covers a step-by-step guide of contributing. It will tell you how to start by setting up the code repository and creating a Pull Request step by step. This guide is modified from [OpenMMlab](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md)'s contribution guide.

### 1. Fork and clone

If you are posting a pull request for the first time, you should fork the PyPose repository by clicking the **Fork** button in the top right corner of the GitHub page, and then the forked repositories will appear under your GitHub profile.

Then, you can clone the repositories to local:

```shell
git clone https://github.com/{your_github_id}/pypose.git
# or if you have set up the SSH-key based authentication, clone by:
# git clone git@github.com:{your_github_id}/pypose.git
```

After that, you should add the official repository as the upstream repository

```bash
git remote add upstream https://github.com/pypose/pypose.git
# or the following if you have cloned using SSH-key based authentication
# git remote add upstream git@github.com:pypose/pypose.git
```

Check whether the remote repository has been added successfully by `git remote -v`. The output should be like this:
```bash
origin	https://github.com/{your_github_id}/pypose.git (fetch)
origin	https://github.com/{your_github_id}/pypose.git (push)
upstream	https://github.com/pypose/pypose.git (fetch)
upstream	https://github.com/pypose/pypose.git (push)
```
or
```bash
origin	git@github.com:{your_github_id}/pypose.git (fetch)
origin	git@github.com:{your_github_id}/pypose.git (push)
upstream	git@github.com:pypose/pypose.git (fetch)
upstream	git@github.com:pypose/pypose.git (push)
```

> Here's a brief introduction to origin and upstream. When we use "git clone", we create an "origin" remote by default, which points to the repository cloned from. In the above example, "origin" is your own forked repo. As for "upstream", we add it ourselves to point to the target repository, which is PyPose's official repo `pypose/pypose`. Of course, you could name it to any arbitrary name other than `upstream`. Usually, contributors will push the code to "origin". If the pushed code conflicts with the latest code in official("upstream"), the contributor should pull the latest code from upstream to resolve the conflicts, and then push to "origin" again. The posted Pull Request will be updated automatically.

### 2. Create a development branch
After cloning a local copy of the repo, we should create a branch based on the main branch to develop the new feature or fix the bug. The proposed branch name is `username/pr_name`

```shell
git checkout -b {your_github_id}/{pr_name}
```

In a subsequent development, if the main branch of the local repository is behind the main branch of "upstream", we need to pull the upstream for synchronization, and then execute the above command:

```shell
git pull upstream main
```

### 3. Commit the code and pass the unit test
- The committed code should pass through the unit test

  ```shell
  # Pass all unit tests
  pytest
  ```

- If the documents are modified/added, we should check the rendering result referring to [guidance](#documentation-rendering)

### 4. Push the code to remote

We could push the local commits to remote after passing through the check of the unit test. You can associate the local branch with your remote branch by adding `-u` option.

```shell
git push -u origin {branch_name}
```

This will allow you to use the `git push` command to push code directly next time, without having to specify a branch or the remote repository.

### 5. Create a Pull Request

(1) Create a pull request in GitHub's Pull request interface

(2) Modify the PR description accordingly so that other developers can better understand your changes

**note**

(a) The Pull Request description should contain the reason for the change, the content of the change, and the impact of the change and be associated with the relevant Issue (see [documentation](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue))

(b) Check whether the Pull Request passes through the CI

<img width="909" alt="image" src="https://user-images.githubusercontent.com/8695500/222796987-3f612b21-21c9-4bb4-b8ff-4b1ddccf115d.png">

CI will run the unit test for the posted Pull Request in the Linux environment. We can see the specific test information by clicking `Details` in the above image so that we can modify the code.

(3) If the Pull Request passes the CI, then you can wait for the review from other developers. You'll modify the code based on the reviewer's comments, and repeat the steps [3](#3-commit-the-code-and-pass-the-unit-test)-[4](#4-push-the-code-to-remote) until all reviewers approve it. Then, we will merge it ASAP.

### 6. Resolve conflicts

If your local branch conflicts with the latest main branch of "upstream", you'll need to resolve them. There are two ways to do this:

```shell
git fetch --all --prune
git rebase upstream/main
```

or

```shell
git fetch --all --prune
git merge upstream/main
```

If you are very good at handling conflicts, then you can use rebase to resolve conflicts, as this will keep your commit logs tidy. If you are not familiar with `rebase`, then you can use `merge` to resolve conflicts.


[//]: # (######################)
## Documentation Rendering
### 1. Contributing to Documentation

### 1.1 Build docs locally

1. Sphinx docs come with a makefile build system. To build PyPose documentation locally, run the following commands. Note that if you are on a clean (newly installed) Ubuntu, you may need to run `sudo apt install build-essential` before the following commands, but it is not needed for MacOS.

```bash
pip install -r requirements/docs.txt
cd docs
make html
```

2. Then open the generated HTML page: `docs/build/html/index.html`.

3. To clean and rebuild the doc:
```bash
make clean
```


### 1.2 Writing documentation

1. For the most simple case, you only need to edit the Python files and add docstring to functions following [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

2. Sometimes you may need to edit rst files like [lietensor.rst](docs/source/lietensor.rst), e.g., adding a new doc page.
More details can be found at [rst markdown](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

3. Commit your changes.
