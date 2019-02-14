First of all, you should have ```python3```, ```git``` and ```virtualenv``` to work with this project.

## How to deploy it:

Note: If you use Windows, you should start a command line (which has python3 in PATH) with administrator's privileges.

#### First of all, clone this repository:
```
git clone https://github.com/greylord1996/MAP.git
```

```
cd MAP/
```

#### Create a new virtual environment
```
python3 -m venv mapenv
```

#### Activate the environment
```
on Linux: source mapenv/bin/activate
on Windows: mapenv/Scripts/activate
```

#### Install required dependencies
```
pip install --upgrade pip -r requirements.txt 
```

#### Finally, run *src/main.py* using python3
```
python3 src/main.py
```

