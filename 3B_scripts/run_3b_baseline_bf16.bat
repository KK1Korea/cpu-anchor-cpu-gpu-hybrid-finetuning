@echo off
echo ============================================================
echo  Project Jellyfish - 3B Baseline bf16 100%%
echo  "The industry standard. What everyone uses."
echo ============================================================
echo.

cd /d C:\LlamaFactory
call venv\Scripts\activate

echo [%date% %time%] Starting 3B Baseline bf16 training...
llamafactory-cli train jellyfish_3b_baseline_bf16.yaml

echo.
echo [%date% %time%] Training complete.
echo ============================================================
pause
