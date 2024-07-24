import argparse
import os
import re
import shutil
import subprocess


# Run the called script with arguments


def list_pt_files(directory):
    """
    Lists all the files in the given directory that end with the `.pt` extension.

    Args:
        directory (str): The path to the directory to search.

    Returns:
        list: A list of file paths that end with the `.pt` extension.
    """
    pt_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            pt_files.append(os.path.join(directory, filename))
    return pt_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default=None,
        required=True,
        help="model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--folder_path",
        type=str,
        default=None,
        required=True,
        help="Path to input image to edit.",
    )
    parser.add_argument(
        "--source_text",
        type=str,
        default=None,
        help="The source text describing the input image.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--n_hiper",
        type=int,
        default=5,
        help="Number of hiper embedding",
    )
    parser.add_argument(
        "--emb_train_steps",
        type=int,
        default=1500,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--emb_learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizing the embeddings.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    args = parser.parse_args()
    img_path = os.path.join(args.folder_path, "images")
    all_img = sorted(
        [os.path.join(img_path, f) for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    )
    output_path = os.path.join(args.folder_path, "ckpt_tmp")
    tgt_text = args.source_text
    for i in range(len(all_img)):
        print(f"[INFO] processing image {i}/{len(all_img)}")
        cur_img = all_img[i]
        cur_out = os.path.join(output_path, f"ckpt_{i}")
        subprocess.run(
            [
                "python",
                "train.py",
                "--pretrained_model_name",
                args.pretrained_model_name,
                "--input_image",
                cur_img,
                "--output_dir",
                cur_out,
                "--seed",
                str(args.seed),
                "--target_text",
                tgt_text,
                "--source_text",
                args.source_text,
                "--n_hiper",
                str(args.n_hiper),
                "--emb_train_steps",
                str(args.emb_train_steps),
                "--emb_learning_rate",
                str(args.emb_learning_rate),
                "--gradient_accumulation_steps",
                str(args.gradient_accumulation_steps),
            ]
        )

    n_images = len(all_img)
    final_ckpt = os.path.join(args.folder_path, "ckpt")
    os.makedirs(final_ckpt, exist_ok=True)
    prefix = "target_pt_{}.pt"
    for i in range(n_images):
        cur_out = os.path.join(output_path, f"ckpt_{i}")
        selected_list = list_pt_files(cur_out)
        selected_list.sort(key=lambda x: int(re.findall(r"\d+", x.split("/")[-1])[0]))
        selected_file = selected_list[-1]
        cur_part = prefix.format(i)
        shutil.copy2(selected_file, os.path.join(final_ckpt, cur_part))
