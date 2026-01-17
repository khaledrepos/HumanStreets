@echo off
start "Backend" cmd /k "cd backend && call ..\..\..\..\.venv\Scripts\activate && uv run uvicorn main:app --reload"
start "Frontend" cmd /k "cd frontend && npm run dev"
echo Demo started in new windows.
