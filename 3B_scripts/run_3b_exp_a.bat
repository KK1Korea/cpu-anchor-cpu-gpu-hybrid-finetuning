@echo off
echo ============================================
echo  Project Jellyfish 🪼 — 3B Experiment A
echo  fp32 (20%%) → bf16 (80%%)
echo ============================================

echo.
echo [%date%:%time%] Phase 1: GPU fp32 anchor (100 steps)...
call C:\LlamaFactory\venv\Scripts\activate
cd /d C:\LlamaFactory
llamafactory-cli train jellyfish_3b_exp_a_phase1.yaml

echo.
echo [%date%:%time%] Phase 1 complete. Starting Phase 2...
echo.
echo [%date%:%time%] Phase 2: GPU bf16 exploration (400 steps)...
llamafactory-cli train jellyfish_3b_exp_a_phase2.yaml

echo.
echo [%date%:%time%] Training complete.
pause
