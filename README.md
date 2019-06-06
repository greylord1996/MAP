There are installers for Windows and Ubuntu/Debian ([see releases](https://github.com/greylord1996/MAP/releases)). You can just download appropriate installer and run it ignoring all steps below.

## How to deploy it:

First of all, you should have ```python 3.x``` to work with this project. The recommended version of Python is ```3.6```.

*Note for Windows users*: You should start a command line with administrator's privileges.

First of all, clone the repository:

    git clone https://github.com/greylord1996/MAP.git
    cd MAP/

Create a new virtual environment:

    # on Linux:
    python3 -m venv mapenv
    # on Windows:
    python -m venv mapenv

Activate the environment:

    # on Linux:
    source mapenv/bin/activate
    # on Windows:
    call mapenv\Scripts\activate.bat

Install required dependencies:

    # on Linux:
    pip install --upgrade pip -r requirements.txt
    # on Windows:
    python -m pip install --upgrade pip -r requirements.txt

Finally, run *src/main.py*:

    # on Linux:
    python3 src/main.py
    # on Windows:
    python src\main.py

