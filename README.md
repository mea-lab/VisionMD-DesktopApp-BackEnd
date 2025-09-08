# VisionMD BackEnd

This repository contains the **back-end** of **VisionMD**, a tool for quantifying motor symptoms from videos. A static version of the front-end is included here, so any changes to both front-end and back-end should be made on the main branch:

GitHub Repository:  
https://github.com/mea-lab/VisionMD-Gait-BackEnd

---

## Prerequisites

- Anaconda (or Miniconda)  
- Python 3.10

---

## Local Setup

### 1. Clone the Repository and Switch to "static" branch

```bash
git clone https://github.com/mea-lab/VisionMD-Gait-BackEnd.git
cd VisionMD-Gait-BackEnd
git checkout static
```

### 2. Create and Activate the Conda Environment

Use the provided `environment.yml` file to recreate the exact development environment:

```bash
conda env create -f environment.yml
conda activate VisionMD
```

### 3. Start the Django Development Server

```bash
python manage.py runserver
```

---

## Open the Application

In your browser (Chrome is recommended), navigate to:  
[http://localhost:8000/](http://localhost:8000/)

---

## Stop the Server

To stop the development server, press `Ctrl + C` in the terminal where the server is running.

---
