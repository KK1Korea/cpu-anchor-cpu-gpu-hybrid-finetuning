@echo off
cd /d C:\LlamaFactory
call venv\Scripts\activate

echo [Exp G] Phase 1 - CPU fp32 anchor, 100 steps
set CUDA_VISIBLE_DEVICES=-1
call llamafactory-cli train jellyfish_exp_g_phase1.yaml
if %ERRORLEVEL% neq 0 (
    echo Phase 1 FAILED
    pause
    exit /b 1
)

echo [Exp G] Phase 2 - GPU fp32 explore, 400 steps
set CUDA_VISIBLE_DEVICES=0
call llamafactory-cli train jellyfish_exp_g_phase2.yaml
if %ERRORLEVEL% neq 0 (
    echo Phase 2 FAILED
    pause
    exit /b 1
)

echo [Exp G] DONE
pause
