# 1. Use a lightweight Python base
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy only the requirements first (this speeds up builds)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your code and models
COPY src/ ./src/
COPY models/ ./models/

# 6. Expose the port FastAPI uses
EXPOSE 8000

# 7. Start the API
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]