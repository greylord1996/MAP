## Manage dependencies:

First of all, you should have ```python 3.x``` to work with this project. The recommended Python version is ```3.6``` or greater.

*Note for Windows users*: You should start a command line with administrator's privileges.

First of all, clone the repository:

    git clone https://github.com/greylord1996/MAP.git
    cd MAP/

Create a new virtual environment:

    # on Linux:
    python -m venv mapenv
    # on Windows:
    python -m venv mapenv

Activate the environment:

    # on Linux:
    source mapenv/bin/activate
    # on Windows:
    call mapenv\Scripts\activate.bat

Install required dependencies:

    # on Linux:
    pip install -r requirements.txt
    # on Windows:
    python -m pip install -r requirements.txt

Finally, run *src/main.py*:

    # on Linux:
    python src/main.py
    # on Windows:
    python src\main.py

