# Idea generation
python ai_scientist/perform_ideation_temp_free.py \
 --workshop-file "ai_scientist/ideas/SPR.md" \
 --model gpt-4o-2024-05-13 \
 --max-num-generations 5 \
 --num-reflections 5

 # Idea implementation
python launch_scientist_bfts.py \
 --load_ideas "ai_scientist/ideas/SPR.json" \
 --load_code \
 --add_dataset_ref \
 --model_writeup o1-2024-12-17 \
 --model_citation gpt-4o-2024-11-20 \
 --model_review gpt-4o-2024-11-20 \
 --model_agg_plots o3-mini-2025-01-31 \
 --num_cite_rounds 10