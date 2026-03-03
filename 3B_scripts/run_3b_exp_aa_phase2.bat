@echo off
echo ============================================================
echo  Project Jellyfish - 3B Exp AA Phase2 (GPU bf16, resume)
echo  Resuming from checkpoint-100 with continuous lr schedule
echo ============================================================
echo.

cd /d C:\LlamaFactory
call venv\Scripts\activate

echo [%date% %time%] Starting Phase2 (bf16, resume from checkpoint-100)...
llamafactory-cli train jellyfish_3b_exp_aa_phase2.yaml

echo.
echo [%date% %time%] Phase2 complete.
echo ============================================================
pause
