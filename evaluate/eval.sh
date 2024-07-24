RENDER_FOLDER='outputs/tsd/A_high_quality_photo_of_ice_cream_sundae@20240721-230738/save/it50000-test'
OUTPUT_DIR='evaluate_results'
NAME='TSD'
N_PARTICLE=8

python eval.py --render_folder "$RENDER_FOLDER" \
            --output_dir "$OUTPUT_DIR" \
            --name "$NAME" \
            --n_particles $N_PARTICLE \