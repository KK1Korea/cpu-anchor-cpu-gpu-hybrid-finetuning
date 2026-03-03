@echo off
echo ============================================================
echo  Project Jellyfish - 3B Baseline fp32 100%%
echo  "How good is fp32 when you run it the whole time?"
echo ============================================================
echo.

cd /d C:\LlamaFactory
call venv\Scripts\activate

echo [%date% %time%] Starting 3B Baseline fp32 training...
llamafactory-cli train jellyfish_3b_baseline_fp32.yaml

echo.
echo [%date% %time%] Training complete.
echo ============================================================
pause
