FROM python:3.10-slim

COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--reload"]