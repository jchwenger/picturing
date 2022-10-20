import os
import argparse
import numpy as np

from PIL import Image
import PIL


def main(args):
    # Access all PNG files in directory
    all_images = os.listdir(args.dir)
    N = len(all_images)

    # Assuming all images are the same size, get dimensions of first image
    dummy = np.array(Image.open(os.path.join(args.dir, all_images[0])).convert("RGB"))

    # Create a NumPy array of floats to store the average (assume RGB images)
    avg_img = np.zeros(dummy.shape, dtype=np.int32)
    min_img = dummy.copy()
    max_img = dummy.copy()

    # Build up average pixel intensities, casting each image as an array of floats
    for i, img in enumerate(all_images):
        print(f"{i}\r", end="")
        current_img = np.array(
            Image.open(os.path.join(args.dir, img)).convert("RGB"), dtype=np.uint8
        )
        avg_img += current_img
        min_img = np.where(current_img <= min_img, current_img, min_img)
        max_img = np.where(current_img >= max_img, current_img, max_img)
    print()

    avg_img = avg_img.astype(np.float32) / N  # averaging
    # Round values in array and cast as 8-bit integer
    avg_img = np.array(np.round(avg_img), dtype=np.uint8)

    results_dir = f"{args.dir}.results"
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # Generate, save and preview final image
    out_avg = Image.fromarray(avg_img, mode="RGB")
    out_avg.save(os.path.join(results_dir, "average.jpg"))
    out_min = Image.fromarray(min_img, mode="RGB")
    out_min.save(os.path.join(results_dir, "min.jpg"))
    out_max = Image.fromarray(max_img, mode="RGB")
    out_max.save(os.path.join(results_dir, "max.jpg"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""""")

    parser.add_argument(
        "--dir", "-d", type=str, default="data", help="""The data directory"""
    )

    args = parser.parse_args()

    main(args)
