# BIA601 - Intelligent Algorithms Assignment (S25)

## Team Constraint
- Team size must be between 5 and 7 members.

## Project Goal
Optimize e-commerce recommendations using a Genetic Algorithm (GA) based on the provided dataset.

## Dataset
Expected files in dataset folder (`datafromdoctor/` or `HW__Data_S25/`):
- users.xlsx
- products.xlsx
- ratings.xlsx
- behavior.xlsx (or behavior_*.xlsx)

You can also set a custom dataset directory with:
```powershell
$env:BIA_DATA_DIR="C:\path\to\HW__Data_S25"
```

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run API + Web UI
```powershell
uvicorn src.main:app --reload
```
Then open:
- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

## Endpoints
- `POST /train` train model and run GA optimization.
- `GET /recommend/{user_id}?top_k=10` get optimized recommendations.
- `GET /metrics` show baseline vs GA metrics.

## Scientific Reference (2024-2026)
Al Sabri, M.A., Zubair, S. & Alnuhait, H.A. (2026).
Improved prediction on recommendation system by creating a new model that employs Mahout collaborative filtering with content-based filtering based on genetic algorithm methods.
Discover Artificial Intelligence, 6, 20.
https://doi.org/10.1007/s44163-025-00678-y

## Git Requirement
All project files, code changes, and report drafts should be versioned in Git.
