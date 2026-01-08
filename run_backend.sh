
#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend
./backend/venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
