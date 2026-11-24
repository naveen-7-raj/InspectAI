# ==============================================================================
# InspectAI - Windows PowerShell Quickstart & Smoke Test Script
# ==============================================================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " InspectAI Windows PowerShell Quickstart & Verification" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Verify Installation and Artifacts
Write-Host "`n[1/3] Running Python Environment & Artifact Verification..." -ForegroundColor Yellow
python scripts/verify_installation.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Installation verification failed. Aborting." -ForegroundColor Red
    exit 1
}

# 2. Execute Pytest Automated Test Suite
Write-Host "`n[2/3] Running Full Pytest Test Suite (140 tests)..." -ForegroundColor Yellow
pytest tests/ -v --durations=5
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Test suite failed. Review failures above." -ForegroundColor Red
    exit 1
}

# 3. Execute WebSocket Replay Smoke Test
Write-Host "`n[3/3] Running WebSocket & REST Prediction Smoke Test..." -ForegroundColor Yellow
python notebooks/verify_stream.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Streaming smoke test failed." -ForegroundColor Red
    exit 1
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " ALL SMOKE TESTS PASSED SUCCESSFULLY! InspectAI is fully operational." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "To launch the platform locally, open two terminals and run:"
Write-Host "  Terminal 1 (Backend):   uvicorn backend.main:app --host 0.0.0.0 --port 8000" -ForegroundColor White
Write-Host "  Terminal 2 (Dashboard): streamlit run frontend/app.py --server.port 8501" -ForegroundColor White

