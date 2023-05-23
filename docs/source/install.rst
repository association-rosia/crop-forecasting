==================
Installation Guide
==================

Conda environement
==================

This guide provides step-by-step instructions for installing the Conda environment. This allows you to recreate the environment to run our project with all its dependencies in a consistent and reproducible manner.

Prerequisites
-------------

Before installing the Conda environment, make sure you have the following prerequisites:

- Conda: If you don't have Conda installed already, you can download and install it from the official `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `Anaconda <https://www.anaconda.com/products/distribution>`_ website.

Installation Steps
------------------

Follow these steps to install the `crop-forecasting-env` Conda environment:

- Open a terminal or command prompt.

- Navigate to the main project directory named `crop-forecasting`.

- Run the following command to create a Conda environment from the `environment.yml` file:

.. code-block:: bash

    conda env create -f environment.yml

- Activate the Conda environment:

.. code-block:: bash

    conda activate crop-forecasting-env



Weights & Biases Account
========================

Overview
--------

Weights & Biases (wandb) is a powerful platform for tracking, visualizing, and analyzing machine learning experiments. To get started with wandb, you'll need to create an account on the wandb website. Here are the steps to create a wandb account:

Go to the WandB website
-----------------------

Open a web browser and navigate to the `wandb <https://wandb.ai/>`_ website.

Click on "Get Started for Free"
-------------------------------

Click on the "Get Started for Free" button on the wandb homepage to begin the account creation process.

Sign Up with a Provider or Email
--------------------------------

You can sign up for a wandb account using your existing Google, GitHub, or email account. Choose the option that suits you best and follow the prompts to sign up.

Complete the Sign Up Process
----------------------------

If you sign up with a provider like Google or GitHub, you'll be prompted to authorize wandb to access your account information. If you sign up with an email, you'll need to provide some additional information like your name and password.

Verify Your Email (if applicable)
---------------------------------

If you sign up with an email, you may need to verify your email address by clicking on a verification link sent to your email inbox. Follow the instructions in the email to complete the verification process.


Congratulations ! You can now start using our project.
======================================================
See :doc:`tutorial` for futher informations about it.
=====================================================