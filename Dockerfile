FROM python:3.11-slim

WORKDIR /app

# Solo las deps del scraper — sin Playwright, sin ML
COPY requirements-scraper.txt .
RUN pip install --no-cache-dir -r requirements-scraper.txt

# Código fuente + datos de referencia (calendario, stadiums)
COPY scripts/ scripts/
COPY data/reference/liga_calendar_rows.csv data/reference/liga_calendar_rows.csv
COPY data/reference/stadiums.csv data/reference/stadiums.csv

# Aseguramos que los módulos sean importables
COPY scripts/__init__.py scripts/__init__.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "scripts/odds/odds_scheduler.py"]
