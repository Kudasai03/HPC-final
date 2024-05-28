FROM python:3.6-slim

WORKDIR /code

ADD . /code

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 5000

ENV NAME World

CMD ["gunicorn", "app.py", "-b", "0.0.0.0:5000"]
