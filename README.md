# Single Block Autoregressive Text Generation Model
## Project Overview

The Single-Block-Autoregressive-Text-Gen project is centered around a concise GPT-based language model, specifically designed with a single Transformer block that incorporates causal masking in its attention layer. The model training is executed on Kaggle, harnessing the computational capabilities of GPU infrastructure to expedite the training process. The IMDB sentiment classification dataset serves as the training data, enabling the model to generate new movie reviews based on user prompts.

For a seamless user experience, a user-friendly interface has been developed using Streamlit. This interface allows users to effortlessly input prompts and receive generated movie reviews. The use of Kaggle's GPU infrastructure for training, combined with the interactive Streamlit interface, creates an efficient and accessible platform for users to explore the autoregressive text generation capabilities of the model.

**NOTE: This repository uses Git Large File Storage (LFS) to handle large files such as the model file. To properly download the model file when cloning this repository, you need to have Git LFS installed on your machine.
If you don't have Git LFS installed, you can install it by following the instructions on the Git LFS website. After installing Git LFS, you can clone this repository as usual with `git clone`.
If you've already cloned the repository without Git LFS, you can download the LFS files with `git lfs pull`.**

# Table of Contents
- [Project Overview](#project-overview)
- [Table of Contents](#table-of-contents)
- [Architecture](#architecture)
- [Installation and Setup](#installation-and-setup)
- [Improvements](#improvements)
- [Acknowledgements](#acknowledgements)

# Architecture

# Installation and Setup
1.  Install Git LFS:
    If you haven't already, download and install Git LFS from https://git-lfs.github.com/.
2. Configure Git LFS:
    Open a terminal and run git lfs install to configure Git LFS for your system.
3. Clone the repository:
    Use the git clone command with the repository URL, including the --recurse-submodules flag to pull large files:

    ```bash
    $ git clone --recurse-submodules https://github.com/<username>/<repo-name>.git
    ```
4. Fetch and download large files:
    Navigate to the cloned repository: cd <repo-name>
    Fetch all objects, including large files: git lfs fetch
    Download the large files:
   
    ```bash
    $ git lfs pull
    ```
6. Verify model download:
    Check if the model file is present in the expected location within the repository.
## Installation
1. Navigate to the directory containing your Streamlit app.
2. Install the required packages using pip:

    ```bash
    $ pip install -r requirements.txt
    ```
3. Run the Streamlit app:

    ```bash
    $ streamlit run app.py
    ```

    The Streamlit app is now running and can be accessed at http://localhost:5000 in your web browser.

    ![streamlit interface](https://github.com/ajinkyavbhandare/Single-Block-Autoregressive-Text-Gen/blob/main/images/app.png)

 
 
 This Streamlit app showcases a compact GPT-based autoregressive language model.  
 [visit here](https://app-z3qbiyomda-uc.a.run.app/)  
 [Kaggle notebook for model training](https://www.kaggle.com/code/ajinkyabhandare2002/single-block-autoregressive-text-gen)  



 ## Downloading the Data

 run download-data.py

 ## Acknowledgements
1. [Text generation with a miniature GPT](https://keras.io/examples/generative/text_generation_with_miniature_gpt/)
