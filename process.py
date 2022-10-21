import os
import argparse
import numpy as np

import PIL
from PIL import Image

# JCW hacking...
# Main inspiration:
# https://stackoverflow.com/a/17383621


def main(args):
    # Access all PNG files in directory
    all_images = [x for x in os.listdir(args.dir) if not x.startswith(".")]
    N = len(all_images) - args.chunksize

    # Assuming all images are the same size, get dimensions of first image
    dummy = np.array(Image.open(os.path.join(args.dir, all_images[0])).convert("RGB"))

    # Create a NumPy array of floats to store the average (assume RGB images)
    def initialize_arrays():
        return (
            np.zeros(dummy.shape, dtype=np.int64),
            np.zeros(dummy.shape, dtype=np.uint8) + 255,
            np.zeros(dummy.shape, dtype=np.uint8),
        )

    def save_to_tiff(arr, name):
        tmp_img = Image.fromarray(arr, mode="RGB")
        tmp_img.save(os.path.join(results_dir, name))


    def save_bulk(data):
        for arr, name in data:
            save_to_tiff(arr, name)

    results_dir = f"{args.dir}.results"
    if not os.path.isdir(results_dir):
        print(f"creating results directory {results_dir}")
        os.mkdir(results_dir)

    # Build up average pixel intensities, casting each image as an array of floats
    for i in range(N):
        print(f"{i}/{N}")
        avg_img, min_img, max_img = initialize_arrays()
        subset = all_images[i:i+args.chunksize]
        for j, img_filename in enumerate(subset):
            print(f"{j}/{len(subset)}\r", end="")
            current_img = np.array(
                Image.open(os.path.join(args.dir, img_filename)).convert("RGB"), dtype=np.uint8
            )
            avg_img += current_img
            min_img = np.where(current_img <= min_img, current_img, min_img)
            max_img = np.where(current_img >= max_img, current_img, max_img)

        avg_img = avg_img.astype(np.float32) / args.chunksize  # averaging
        # Round values in array and cast as 8-bit integer
        avg_img = np.array(np.round(avg_img), dtype=np.uint8)

        save_bulk((
            [avg_img, f"average.{i}.tiff"],
            [min_img, f"min.{i}.tiff"],
            [max_img, f"max.{i}.tiff"],

        ))
        print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""""")

    parser.add_argument(
        "--dir", "-d", type=str, default="data", help="""The data directory"""
    )

    parser.add_argument(
        "--chunksize", "-c", type=int, default=90, help="""
        The chunk size on which the sliding calculation will be performed
        """
    )

    args = parser.parse_args()

    main(args)
