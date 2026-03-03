@echo off
echo ============================================================
echo  Project Jellyfish - 3B Exp CPU-AA Phase1 (CPU fp32)
echo  *** IMPORTANT: Press Ctrl+C AFTER step 100 appears! ***
echo  checkpoint-100 is auto-saved. Stop anytime after step 100.
echo  Expected time: ~1-2 hours for 100 steps on CPU
echo ============================================================
echo.

cd /d C:\LlamaFactory
call venv\Scripts\activate

set CUDA_VISIBLE_DEVICES=-1

echo [%date% %time%] Starting Phase1 (CPU fp32, 500-step cosine)...
llamafactory-cli train jellyfish_3b_exp_cpu_aa_phase1.yaml

echo.
echo [%date% %time%] Phase1 complete.
echo ============================================================
pause
