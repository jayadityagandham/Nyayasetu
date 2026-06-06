@echo off
echo Starting NyayaSetu API Backend...
start cmd /k "pip install -r requirements.txt && pip install fastapi uvicorn python-multipart && uvicorn api:app --host 0.0.0.0 --port 8000"

echo Starting Next.js Frontend...
cd frontend
start cmd /k "npm install && npm run dev"

echo Both services are starting...
echo Next.js will be available at http://localhost:3000
echo FastAPI will be available at http://localhost:8000
