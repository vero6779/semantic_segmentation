# Semantic Segmentation Deployment Scaffold

## Overview
This project provides a complete, production-ready scaffold for deploying a Semantic Segmentation model using Streamlit Cloud. It's built with modularity and scalability in mind, separating the predictor logic, mask utilities, and UI.

## Semantic vs Instance Segmentation
- **Semantic Segmentation** assigns a class label to every pixel in an image. It does not distinguish between different instances of the same object class. For example, two overlapping cars would be segmented as one continuous blob of "car".
- **Instance Segmentation** detects and delineates each distinct object of interest in an image, allowing you to distinguish between "car 1" and "car 2".

## Running Locally

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Test the inference engine locally (ensure you have an `assets/demo.png`):
   ```bash
   python app.py
   ```
3. Run the Streamlit web application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Streamlit Cloud Deployment Guide

1. Push this repository to GitHub.
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your_repo_url>
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select your repository and deploy `streamlit_app.py`.
4. Streamlit Cloud runs on Debian Linux. The `packages.txt` file includes `libgl1` which is crucial to avoid `libGL.so.1` missing errors when using OpenCV.
5. Missing modules (`ModuleNotFoundError`) are typically solved by ensuring your `requirements.txt` is updated and committed.
