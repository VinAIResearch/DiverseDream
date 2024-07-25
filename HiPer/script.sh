# Create folder "images" under OUT_PATH and put all reference images under this folder
MODEL_PATH="stable-diffusion-v1-4"
OUT_PATH="output_hiper/A_high_quality_photo_of_an_ice_cream_sundae"
PROMPT="A high quality photo of ice cream sundae"
python multi_hiper.py \
--pretrained_model_name "$MODEL_PATH" \
--folder_path "$OUT_PATH" \
--source_text "$PROMPT" \
--n_hiper=5 \
--emb_learning_rate=5e-3 \
--emb_train_steps=1500 \
--seed 200000