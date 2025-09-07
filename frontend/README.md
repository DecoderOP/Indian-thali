# Personalised Indian Thali - Āyush Thālī

A web application that recommends personalized Indian thali (meal plates) based on user symptoms and dietary preferences, focused on nutritional wellness.

## Project Structure

- `/app` - Next.js frontend application
- `/backend` - Python Flask API for nutritional analysis and recommendations
- `/lib` - Shared TypeScript types and utilities
- `/public` - Static assets

## Getting Started

### Prerequisites

- Node.js (v18+)
- Python (v3.8+)
- npm or yarn
### Installation

1. Clone this repository
2. Install frontend dependencies:
   ```bash
   npm install
   ```
3. Install backend dependencies:
   ```bash
   cd backend
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   pip install -r requirements.txt
   ```

### Running the Application

For development, you can use the provided batch file (Windows) to run both frontend and backend:

```bash
run-dev.bat
```

Or manually:

1. Start the Python backend:
   ```bash
   cd backend
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   python app.py
   ```

2. In a separate terminal, start the Next.js frontend:
   ```bash
   npm run dev
   ```

3. Open your browser and navigate to http://localhost:3000

## Features

- Chat interface to describe symptoms
- Nutritional analysis based on user input
- Personalized thali recommendations
- Support for dietary preferences (vegetarian, vegan, gluten-free)

## Technology Stack

- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **Backend**: Python, Flask
- **API**: RESTful JSON API

## Environment Variables

### Frontend (.env.local)

```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### Backend (.env)

```
PORT=5000
FLASK_ENV=development
```

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
