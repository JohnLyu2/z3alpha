FROM python:3.12-slim

WORKDIR /api-flask

COPY webapp/ /api-flask/webapp/
COPY wsgi.py requirements.txt  /api-flask/
COPY smtlib2025.sqlite /api-flask/

RUN pip3 install --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "wsgi:app", "-e", "SMTLIB_DB=/api-flask/smtlib2025.sqlite", "-b", "0.0.0.0:5000", "-w", "4"]
