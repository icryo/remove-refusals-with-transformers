# One-time: install uv and clone
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/icryo/remove-refusals-with-transformers/ householder && cd householder

# 1. Edit the model (produces ./edited_qwen)
./hh.py --model_id Qwen/Qwen1.5-7B-Chat \
        --output_path ./edited_qwen \
        --harmful harmful.txt \
        --harmless harmless.txt

# 2. Evaluate the edit
./hh_eval_fast.py \
        --base_id  Qwen/Qwen1.5-7B-Chat \
        --edited   ./edited_qwen \
        --harmful  harmful.txt \
        --harmless harmless.txt \
        --n 10 --batch_size 4
