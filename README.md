First of all you need to unzip files inside data1.zip into the root folder

1. create venv:
```sh
python -m venv venv
```
2. activate venv:
```sh
venv/Scripts/activate
```
3. install dependencies:
```sh
pip install -r requirements.txt
```
4. start application:
```sh
uvicorn test:app
```
5. application started:
```
it is located on localhost:8000
```