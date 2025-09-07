# Indian Thali 🍲

AI-powered personalized diet planning system that generates customized Indian
diet plans based on user health data.

## Project Overview

- **Frontend**: Next.js application with TypeScript
- **Backend**: Python FastAPI application
- **Integration**: REST API communication between frontend and backend

## Setup

### Backend Setup

1. Create and activate a Python virtual environment:

   ```
   cd backend
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On Unix/MacOS
   ```

2. Install backend dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the backend server:

   ```
   cd backend
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   Or use the provided script:

   ```
   run-backend.bat
   ```

### Frontend Setup

1. Install frontend dependencies:

   ```
   cd frontend
   npm install
   ```

2. Run the frontend development server:

   ```
   cd frontend
   npm run dev
   ```

   Or use the provided script:

   ```
   cd frontend
   run-dev.bat
   ```

### Run Both Together

You can run both servers simultaneously using:

```
run-all.bat
```

This will start:

- Backend at http://localhost:8000
- Frontend at http://localhost:3000

## API Endpoints

- `GET /`: Root endpoint to check if API is running
- `POST /api/generate-plan`: Generate a personalized diet plan

## Notes

- Do not commit `.env` or `venv/` directories
- Backend uses FastAPI with CORS enabled for frontend communication
- Frontend fetches data from the backend API and displays the results
