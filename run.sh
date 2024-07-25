PROMPT="A high quality photo of ice cream sundae"
HIPER_PATH="output_hiper/A_high_quality_photo_of_an_ice_cream_sundae/hiper_5/ckpt"

# VSD with HiPer
python launch.py --config configs/tsd.yaml --train --gpu 0 system.prompt_processor.prompt="$PROMPT" system.n_particles=8 system.prompt_processor.hiper_scale=0.9 system.guidance.token_len=8 system.hiper_path="$HIPER_PATH" name="vsd_w_hiper" system.guidance.use_lora=true

# TSD
python launch.py --config configs/tsd.yaml --train --gpu 0 system.prompt_processor.prompt="$PROMPT" system.n_particles=8 system.prompt_processor.hiper_scale=0.9 system.guidance.token_len=8 system.hiper_path="$HIPER_PATH" system.guidance.use_lora=false