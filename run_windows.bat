@ECHO OFF
set ENV_NAME=image-caption-env

if [%ENV_NAME%]==[] goto "VIRTUAL_ENVIRONMENT_NAME"
set ENV_PATH="%root%\user_environments\%ENV_NAME%"

If Not Exist %ENV_PATH% (
    	ECHO Creating the Virtual Environment:  %ENV_NAME%
	py -m venv %ENV_PATH% || goto :error
	ECHO SUCESS: Virtual Environment Created on %ENV_PATH%


)

ECHO Activating the Created Environment...
CALL %ENV_PATH%\Scripts\activate.bat || goto :error
ECHO SUCCESS: Virtual Environment ACTIVATED

ECHO Checking Dependencies...
	py -m pip install --upgrade pip || goto :error
	py -m pip install -r requirements.txt || goto :error
	py -m pip install --upgrade setuptools wheel || goto :error
	py -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
ECHO SUCCESS: Dependencies Installed

nvcc --version
ECHO Running App....
streamlit run src\main.py -- --disable_training 0



:error
echo Failed with error.
pause
exit /b %errorlevel%
