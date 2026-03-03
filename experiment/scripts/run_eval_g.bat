@echo off
cd /d C:\LlamaFactory
call venv\Scripts\activate

echo [Eval G] MMLU subset, 200 samples per task
set CUDA_VISIBLE_DEVICES=0
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,peft=saves/jellyfish_exp_g,trust_remote_code=True,dtype=bfloat16 --tasks mmlu --num_fewshot 5 --batch_size 1 --limit 200 --output_path saves/jellyfish_eval_g

echo [Eval G] DONE
pause
