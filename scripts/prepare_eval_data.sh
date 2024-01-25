mkdir -p data/downloads
mkdir -p data/eval

# GSM dataset
wget -P data/eval/gsm/ https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/test.jsonl

# Codex HumanEval
wget -P data/eval/codex_humaneval https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz

# TruthfulQA
wget -P data/eval/truthfulqa https://github.com/sylinrl/TruthfulQA/raw/main/TruthfulQA.csv

# Toxigen data
mkdir -p data/eval/toxigen
for minority_group in asian black chinese jewish latino lgbtq mental_disability mexican middle_east muslim native_american physical_disability trans women
do
    wget -O data/eval/toxigen/hate_${minority_group}.txt https://raw.githubusercontent.com/microsoft/TOXIGEN/main/prompts/hate_${minority_group}_1k.txt
done
