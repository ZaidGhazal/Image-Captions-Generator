#!/usr/bin/env python
import subprocess
import sys, os
import webbrowser


def main():

    # Getting path to python executable 
    executable = sys.executable

    # Open browser tab. May temporarily display error until streamlit server is started.
    webbrowser.open("http://localhost:8501")

    # Run streamlit server
    file_directory = os.path.realpath(__file__).split("main.py")[0]
    path_to_app = os.path.join(file_directory, "app.py")
    path_to_app = '"' + path_to_app + '"' 
    
    result = subprocess.run(
        f"{executable} -m streamlit run {path_to_app} --server.headless=true --global.developmentMode=false",
        shell=True,
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    main()