# CounterPass

**CounterPass** is an AI-based football performance review and passing decision analysis system. It is designed to take match video footage, analyze player and ball tracking using temporal computer vision, map players onto pitch coordinates, and evaluate passing options, lanes, missed passes, and player performance metrics.

This system addresses the academic challenge that passing options cannot always be determined by single-frame analysis due to occlusion, movement, and changing lane states. It utilizes a temporal window analysis structure (velocity tracking, trajectory predictions, and historical evidence clustering).

---

## Technical Stack
- **Frontend**: React, Vite, Tailwind CSS, Recharts, Lucide React
- **Backend**: FastAPI, Uvicorn, SQLAlchemy ORM, SQLite
- **Computer Vision**: OpenCV, NumPy, SciPy
- **Testing**: pytest

---

## Project Structure
- `backend/`: FastAPI server, database configuration, OpenCV analytical models, and SQLite engine.
- `frontend/`: React application built with Vite and styled with Tailwind CSS.

---

## Installation & Setup

Detailed configuration guides will be added as we proceed through the development phases.
