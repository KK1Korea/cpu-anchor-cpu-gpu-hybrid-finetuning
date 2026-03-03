@echo off
cd /d C:\LlamaFactory
call venv\Scripts\activate

echo ============================================================
echo  Jellyfish Exp A : GPU-only 500 steps (Baseline)
echo  QLoRA 4-bit, bf16, Qwen2.5-7B-Instruct
echo  Output: saves/jellyfish_exp_a/
echo ============================================================
echo.

set CUDA_VISIBLE_DEVICES=0
llamafactory-cli train jellyfish_exp_a.yaml

echo.
echo ============================================================
echo  Experiment A Done
echo  Result: saves/jellyfish_exp_a/trainer_state.json
echo ============================================================

pause
