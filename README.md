# VisionMD BackEnd

This repository holds the source code for the backend of the VisionMD Desktop App. Below is documentation for developing, building and testing the backend server of VisionMD.

## Prerequisites
- Anaconda (or Miniconda)  
- Python 3.10

## Development
Follow the below steps to get the server running for development.

### 1. Clone the Repository

```bash
git clone https://github.com/mea-lab/VisionMD-DesktopApp-BackEnd.git
cd VisionMD-DesktopApp-BackEnd
```

### 2. Create and Activate the Conda Environment

Use the provided `environment.yml` file to recreate the exact development environment. Make sure you use the environment according to your OS.

```bash
conda env create -f environment_{OS}.yml
conda activate VisionMD
```

### 4. Download the models
Download the models using the scripts found in `./scripts`. 
```bash
./scripts/get_models.sh # For Linux / MacOS
./scripts/get_models.bat # For Windows
```

### 3. Start the Django Development Server

```bash
python manage.py runserver
```
You must have the backend Django server running on `127.0.0.1:8000`. After, run the frontend repository located at https://github.com/mea-lab/VisionMD-DesktopApp-FrontEnd. Follow the instruction in the README to run the frontend for developement. After setup, the frontend will now connect to your backend.

### Stop the Server
To stop the development server, press `Ctrl + C` in the terminal where the server is running.

## Testing static web assets
This documents internal testing for the VisionMD Desktop App using static web assets.

### Build and transfer the static web asssets
Follow the README in https://github.com/mea-lab/VisionMD-DesktopApp-FrontEnd to build the static web assets. After transferring the static web assets to the root of this repository, rename it to `dist`.

### Start the server
```bash
python manage.py runserver
```

### Open the Application

In your browser (Chrome is recommended), navigate to:  
[http://localhost:8000/](http://localhost:8000/). The app will be available within the browser.

## Building for Production
This section documents building the Pyinstaller executable for the production installers for Windows, Linux and MacOS. The Pyinstaller executable has to be packaged with the frontend repository to create a proper installer.

### Building the executable
```bash
./scripts/build_windows.sh # For Windows
./scripts/build_linux.sh # For Linux
./scripts/build_mac.sh # For MacOS
```
Run the appropriate script for your OS. This will build a onedir Pyinstaller executable at `./pyinstaller_builds/serve_{OS}.` Transfer this executable to the `./pyinstaller_builds/` directory in the frontend repository. The frontend repository is now ready for building a production installer.