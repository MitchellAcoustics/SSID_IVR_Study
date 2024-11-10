# Soundscape Audio-Visual Interactions in Virtual Reality

## Introduction

This project is in a draft state.

## Instructions for Authors

### Setup

1. Install the following tools:

   - VS Code. Also install the following extensions:
     - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
     - [Juptyer](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
     - [Quarto](https://marketplace.visualstudio.com/items?itemName=quarto-dev.quarto-vscode)
   - [uv](https://docs.astral.sh/uv/)
   - [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)
   - [Github Desktop](https://desktop.github.com/)

     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

   - [Quarto](https://quarto.org/docs/get-started/)
     - Once installed, run the following command to install LaTeX dependencies if you don't have it already:

       ```bash
       quarto install tinytex
       ```

2. Clone the repo by pressing the green "Code" button and selecting "Open with Github Desktop" (if you're comfortable with Git, do this however you'd like). Select a location on your computer to save the repo - I recommend not saving it in a OneDrive or Google Drive folder.
3. Open the repo in VS Code by clicking "Open in Visual Studio Code" in Github Desktop.
4. **Create a new branch** by clicking the branch name in the bottom left corner of the VS Code window and selecting "Create new branch". I recommend naming the branch something like 'Tin-draft'.
5. Open a terminal in VS Code by pressing `Ctrl + ~`. For all VS code commands, you can also press `Cmd + Shift + P` and type the command you want to run.
6. Install the Python dependencies by running the following command in the terminal:

    ```bash
    uv sync
    ```

7. Point VS Code to the Python interpreter by pressing `Ctrl + Shift + P` and typing "Python: Select Interpreter". Select the Python interpreter that has the path `./.venv/Scripts/python` (or `./.venv/bin/python` on MacOS/Linux). If you don't see this option, you may need to restart VS Code.

### Writing

We are using the Quarto markdown format for writing. You can find the documentation [here](https://quarto.org/docs/).

Specifically, we're using a Quarto manuscript project. This means we have one main `.qmd` file that includes the primary text for the paper. Computational outputs (like code and figures) are included in separate `.ipynb` files under the `notebooks` folder and are included in the main manuscript using the `embed` function. You can see an example of this in the `manuscript.qmd` file.

This enables non-code authors to write the main text of the paper in markdown, while coding authors can write code in Jupyter notebooks and include the outputs in the main manuscript.

To preview the manuscript, open the `manuscript.qmd` file and press the Quarto preview button in the top right of VS code (or use the `Cmd + Shift + K` shortcut *or* press `Cmd + Shift + P` and type "Quarto: Preview Manuscript". This will open a preview of the manuscript in VS Code's internal browser.

To do a full render of the manuscript, run the following command in the terminal:

```bash
quarto render
```

This will render the manuscript to a PDF and HTML file in the `output` folder.

### Saving / Committing Changes

1. Save your changes by pressing `Ctrl + S`.
2. Click the "Source Control" icon in the left sidebar of VS Code (it looks like a branch with a checkmark). You should see a list of files that have been changed.
   1. You can review the changes by clicking on the file name. This will open a view with the old version of the file on the left and the new version on the right with changed lines highlighted.
3. Add a commit message in the text box at the top of the Source Control panel. This should be a short description of the changes you made. For example, "Added introduction section".
4. Click the checkmark icon to commit the changes. This saves the changes to your local branch.
5. Push the changes to the remote repository by clicking the three dots next to the checkmark icon and selecting "Push". This will save the changes to the remote repository on Github, on your branch.
6. If you're ready to merge your changes into the main branch, you can create a pull request by clicking the "Create Pull Request" button in Github Desktop. This will open a browser window where you can create a pull request on Github. Select the `draft` branch as the target. You can assign reviewers to review your changes before they are merged.
7. Once the pull request is approved, you can merge the changes by clicking the "Merge" button on Github. This will merge the changes into the `draft` branch.
8. You can delete your local branch by clicking the trash can icon next to the branch name in Github Desktop.
9. If you want to update your local branch with changes from the remote repository, you can click the "Fetch origin" button in Github Desktop, then click the "Update from draft" button to update your local branch with changes from the remote repository.

Once we have a final draft, we can create a release by merging the `draft` branch into the `main` branch. This will trigger a Github action that will render the manuscript to a PDF and HTML file and publish it to the `gh-pages` branch, which will be viewable at `drandrewmitchell.com/SSID_IVR_Study`. 

See an example of what this will look like [here](https://drandrewmitchell.com/J2401_JASA_SSID-Single-Index/).

## Running Video Segmentation

To use `cityseg` to segment a video, edit the config file in the `cityseg_configs` folder to point to the video you want to segment. Then run the following command in the terminal:

```bash
uvx --prerelease=allow cityseg <path_to_config_file>
```

The `uvx` command is a great tool which very quickly downloads the package and all its dependencies and runs it. It's a great way to run a Python script without having to worry about installing dependencies. The `--prerelease=allow` flag is necessary because `cityseg` is only available in prerelease right now.

If you just want to permanently install the package, you can run the following command:

```bash
uv tool install cityseg --prerelease=allow
```

Then you can run the script with the following command:

```bash
cityseg <path_to_config_file>
```
