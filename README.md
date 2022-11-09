First of all you need to unzip files inside data1.zip into the root folder

1. create venv:
	python -m venv venv
2. activate venv:
	venv/Scripts/activate
3. install dependencies:
	pip install -r requirements.txt
4. start application:
	uvicorn test:app
5. application started:
	it is located on localhost:8000