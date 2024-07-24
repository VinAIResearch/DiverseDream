import argparse
import os

from eval_utils import ImageDataset_IQ_IV_new, ImageDataset_IQ_IV_old, extract_dino_features, inception_score_IQ


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval 3D results")
    parser.add_argument(
        "--render_folder",
        type=str,
        default=None,
        required=True,
        help="folder of test phase of threestudio (consist of 120 rendered views)",
    )

    parser.add_argument("--name", type=str, default=None, required=True, help="name of output txt")

    parser.add_argument("--output_dir", type=str, default=None, required=True, help="directory to save results")

    parser.add_argument("--n_particles", type=int, default=6, required=True, help="number of particle")

    args = parser.parse_args()
    n_particles = args.n_particles
    name = args.name
    render_folder = args.render_folder
    os.makedirs(args.output_dir, exist_ok=True)
    file_IQ = os.path.join(args.output_dir, f"IQ_{name}.txt")
    file_IV = os.path.join(args.output_dir, f"IV_{name}.txt")
    file_cosinsim = os.path.join(args.output_dir, f"cosinsim_{name}.txt")

    rendered_ds_iq_iv_new = ImageDataset_IQ_IV_new(path=render_folder, n_particles=n_particles, num_img=120)
    IQ_new, IV_new = inception_score_IQ(rendered_ds_iq_iv_new, device="cuda:0", batch_size=1, splits=1)
    result_iq_iv_new = f"IQ_new = {IQ_new} \n, IV_new = {IV_new}"
    try:
        with open(file_IQ, "a") as file:
            # Iterate through the strings and write them to the file
            file.write(f"{IQ_new[0]}\n")  # Add a newline character to separate lines
            file.flush()
    except Exception as e:
        print(f"An error occurred: {e}")
    try:
        with open(file_IV, "a") as file:
            # Iterate through the strings and write them to the file
            file.write(f"{IV_new}\n")  # Add a newline character to separate lines
            file.flush()
    except Exception as e:
        print(f"An error occurred: {e}")

    print(result_iq_iv_new)

    rendered_ds_iq_iv_old = ImageDataset_IQ_IV_old(path=render_folder, n_particles=n_particles, num_img=120)
    mean_cosine = extract_dino_features(rendered_ds_iq_iv_old, batch_size=1)
    result_cosine_sim = f"cosine sim: {mean_cosine}"
    try:
        with open(file_cosinsim, "a") as file:
            # Iterate through the strings and write them to the file
            file.write(f"{mean_cosine}\n")  # Add a newline character to separate lines
            file.flush()
    except Exception as e:
        print(f"An error occurred: {e}")

    print(result_cosine_sim)
