# Reproducible Research template with Quarto

This is a template repo for generating a manuscript from Quarto and publishing it with a repoducible environment.

## Usage

In general, you can use this repo in the same way recommended by [Quarto](https://quarto.org/docs/manuscripts/authoring/vscode.html), as it is based on their template manuscript repo.

If you are new to Quarto, I would recommend starting with their [Getting Started](https://quarto.org/docs/getting-started.html) guide.

### Setup

Simply click the `Use this template` button to create a new repo based on this template. Then, clone the repo to your local machine and open it in VSCode. You can then start writing your manuscript in the `manuscript.md` file.

This will get you started right away - for more advanced features, see below.

### Manuscript Writing

To write a manuscript, simply edit the `manuscript.md` file. Using the Quarto extension in VSCode, you can preview the manuscript as you write it, or you can use the `quarto render` command to generate the manuscript. By default, this template will render to a PDF using the [Elsevier template](https://github.com/quarto-journals/elsevier), to a Word document, and to an HTML document.

All outputs will be rendered to the `_manuscript` folder by default. You can change this in the `_quarto.yml` file.

Integrating code and their outputs is easy with Quarto. You can include code chunks directly in the manuscript using ` ```{python} ` and render/run the manuscript like a Jupyter notebook. Or, you can offload the code to separate files and include the outputs in the manuscript. This is useful for keeping the manuscript clean and focused on the narrative.

I recommend offloading computationally expensive code to separate `.ipynb` files in the `notebooks` folder and embedding the outputs in the manuscript. This keeps the manuscript cleaner, but more importantly, `.ipynb` notebooks work very well for caching outputs without needing to re-run the code when rendering the manuscript. In theory, Quarto does this with `.qmd` files using `_freeze` and `cache`, but I have found that it is not as reliable as using Jupyter notebooks.

When the HTML version is rendered, code outputs will be included in the main manuscript where they have been embedded, and a link to the source notebook will be included both next to the output and in the table of contents. Clicking on the link will open the notebook in the browser.

By default, the PDF will not include code blocks themselves, but you can include them by setting the `echo` option to `true` in the frontmatter.

### Reproducible Programming Environment

Some additions have been made to improve the reproducibility of the manuscript:

**1. Programming environment from dependency files**: Using the same tools as Binder (i.e. `repo2docker`) this repository is set up to automatically build a programming environment from the `environment.yml`, `requirements.txt`, and/or `install.R` file. Although I would recommned using a standard local setup, this can be useful for working on this repo with others.

**1. Pre-build and cache Binder environment**: Quarto manuscripts are able to easily include a link to a Binder page to allow readers to fully explore the code and data used in the manuscript. However, Binder environments can take a long time to start up, especially with large dependencies. To help with this, we include a Github Action that uses `repo2docker` to build the environment and cache it in the repo. This way, the environment is ready to go when the reader clicks on the Binder link. This action will be triggered to run on any push which changes one of the `repo2docker` configuration files.

**1. DevContainer for local reproducibility**: If you are using VSCode, you can use the included `devcontainer.json` file to create a reproducible environment for local development. This will pull the same environment as the one used in the Binder link, so you can be sure that your code will run in the same environment as the one used to generate the manuscript.

#### Setup for `repo2docker`

To fully set up this functionality, you will need to create a new repository on Github and add the following secrets:

- `DOCKER_USERNAME`: Your Docker username
- `DOCKER_PASSWORD`: Your Docker password

You will also need to change the name of the docker image to use in the `devcontainer.json`. By default, `binder.yml` will build an image with the repository name in lower case (e.g. `mitchellacoustics/reproducible-repository`). To correctly pull this image into the devcontainer, you will need to change the `image` field in the `devcontainer.json` to match the name of the image.

### Publishing

Once you have written your manuscript, you can publish it to a variety of formats. The most common formats are PDF (via LaTeX), Word, and HTML, but you can also publish to several other formats, such JATS and PDF via Typst.

You can publish directly by running `quarto publish gh-pages`^[Or another hosting option such as Quarto Pub or Posit Connect, see [here](https://quarto.org/docs/publishing/)] in the terminal. This will render the manuscript and push the output to the `gh-pages` branch. You can then view the published manuscript at `https://<username>.github.io/<repo-name>`.

By default, the repository is set up with a Github Action file `publish.yml` which will render and publish to the `gh-pages` branch whenever a push is made to the `main` branch. This way, you can simply push your changes to the `main` branch and the manuscript will be automatically updated.

I recommend protecting the `main` branch to limit who and what sort of changes can be pushed to `main`. This will help ensure you can trust the published manuscript and make draft changes without publishing them^[Note: I am looking at the possibility of using `tags`/`releases` to make this more explicit.]. I'd then recommend doing any drafting on a `draft` branch or similar, and then merging to `main` when you are ready to publish.

Each author can have separate branches or forks to work on their own sections of the manuscript, and then merge them into the main branch when they are ready. This can help to avoid conflicts and keep the manuscript clean.

## Errata

### Local Development for `repo2docker`

Instructions and configuration documentation for `repo2docker` can be found [here](https://repo2docker.readthedocs.io/en/latest/config_files.html).

If you use conda, setting this repo up for `repo2docker` should be quite straightforward - simply include an `environment.yml` file with your dependencies.

Since I prefer `uv` and have moved away from conda, here are some further tips:

#### Python

I recommend using `uv` for managing dependencies in Python. However, this can cause issues with the `repo2docker` environment. Right now the solution is to maintain a `pyproject.toml` and `uv.lock` file with `uv`, then run

```bash
uv pip compile pyproject.toml --python-platfom linux --python-version <your-python-version> -o requirements.txt
```

This will generate a `requirements.txt` file that can be used by `repo2docker` to build the environment. You should then include the python version in the `runtime.txt` file as `python-<your-python-version>`.

Note that `repo2docker` currently only supports up to python 3.12, so when setting up your `uv` environment, you may need to specify e.g. `uv python pin 3.12`.

#### R

R dependencies should be defined in an `install.R` file or `DESCRIPTION` file. The R version should be specified in the `runtime.txt` file as `r-<your-r-version>`.

See more info from `repo2docker` [here](https://repo2docker.readthedocs.io/en/latest/config_files.html).

#### Quarto

Quarto is installed using a `postBuild`^[For some reason, the default `postBuild` script includes `bash -v` at the top - the `-v` throws an error for `repo2docker` so it has been removed. Still seems to work though!] script provided by `quarto use binder`^[This command has already been run for this repo. If you are inclined to remove binder compatibility or change the setup, feel free.]. This script is run after the environment has been built and installs Quarto. You can modify this script to install additional dependencies if needed.

### Custom Domain

If you'd like to set a custom domain for the published manuscript, you can do so by adding a `CNAME` file to the `gh-pages` branch. This file should contain the domain you'd like to use, e.g. `example.com`. You can then set up a CNAME record in your DNS settings to point to `<username>.github.io`.

Alternatively, if you set up a Github user site, e.g. for your personal website, with a custom domain, then Github Pages will automatically publish this repo as a subdomain of your user site. For example, if your user site is `example.com`, then the published manuscript will be available at `example.com/repo-name`.
