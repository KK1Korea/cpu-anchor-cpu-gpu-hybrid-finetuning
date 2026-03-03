@echo off
echo ============================================================
echo  Project Jellyfish - 3B Exp AA (continuous lr) MMLU Evaluation
echo ============================================================
echo.

cd /d C:\LlamaFactory
call venv\Scripts\activate

echo [%date% %time%] Starting MMLU evaluation (Exp AA)...
lm_eval --model hf ^
  --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,peft=saves/jellyfish_3b_exp_aa,dtype=bfloat16 ^
  --tasks mmlu ^
  --num_fewshot 5 ^
  --batch_size 1 ^
  --limit 200 ^
  --output_path saves/jellyfish_3b_exp_aa/mmlu_results

echo.
echo [%date% %time%] Evaluation complete.
echo ============================================================
pause
