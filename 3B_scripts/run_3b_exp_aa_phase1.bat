@echo off
echo ============================================================
echo  Project Jellyfish - 3B Exp AA Phase1 (GPU fp32)
echo  *** IMPORTANT: Press Ctrl+C AFTER step 100 appears! ***
echo  checkpoint-100 is auto-saved. Stop anytime after step 100.
echo ============================================================
echo.

cd /d C:\LlamaFactory
call venv\Scripts\activate

echo [%date% %time%] Starting Phase1 (fp32, 500-step cosine)...
llamafactory-cli train jellyfish_3b_exp_aa_phase1.yaml

echo.
echo [%date% %time%] Phase1 complete.
echo ============================================================
pause
