@echo off
cd /d C:\LlamaFactory
call venv\Scripts\activate

echo ============================================================
echo  Jellyfish Exp C : CPU-GPU Hybrid
echo  Phase1: CPU fp32 100 steps  /  Phase2: GPU bf16 400 steps
echo  Output: saves/jellyfish_exp_c/
echo ============================================================
echo.

echo --- Phase 1: CPU Anchor (fp32, 100 steps) ---
echo.

set CUDA_VISIBLE_DEVICES=-1
llamafactory-cli train jellyfish_exp_c_phase1.yaml

if %ERRORLEVEL% neq 0 (
    echo.
    echo Phase 1 FAILED
    pause
    exit /b 1
)

echo.
echo --- Phase 1 Done. Starting Phase 2 ---
echo.

set CUDA_VISIBLE_DEVICES=0
llamafactory-cli train jellyfish_exp_c_phase2.yaml

if %ERRORLEVEL% neq 0 (
    echo.
    echo Phase 2 FAILED
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Experiment C Done (Phase1 + Phase2)
echo  Result: saves/jellyfish_exp_c/trainer_state.json
echo ============================================================

pause
