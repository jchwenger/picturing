import os
import argparse
import numpy as np

import PIL
from PIL import Image

# Jérémie C. Wenger hacking...
# Main inspiration:
# https://stackoverflow.com/a/17383621


def main(args):

    if args.dir.endswith("/"):
        args.dir = args.dir[:-1]

    if args.schedule:
        args.schedule = [
            [int(i) for i in s.split(",")] for s in args.schedule.split(";") if s
        ]

    # Access all PNG files in directory
    all_images = [x for x in os.listdir(args.dir) if not x.startswith(".")]
    N = len(all_images)

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

    i = 0

    print(
        f"chunksize: {args.chunksize} | increment: {args.increment} | step size: {args.step_size}"
    )
    base_increment = args.increment
    base_chunksize = args.chunksize
    base_step_size = args.step_size

    if args.schedule:
        schedule_i = 0
        current_schedule = args.schedule[schedule_i]
        print("schedule")
        for sch in args.schedule:
            print(
                f"- from frame {sch[0]}, until reaching {sch[1]}, increment: {sch[2]} "
            )
        print("-" * 40)
        # direction of increments, up or down
        up = True if args.chunksize < current_schedule[1] else False
        # condition: if we go beyond the targeted chunkside, above or below
        def cond():
            return (
                args.chunksize >= current_schedule[1]
                if up
                else args.chunksize <= current_schedule[1]
            )

    while i < N - args.chunksize:

        if args.schedule:

            if i == current_schedule[0]:
                print(
                    f"Entering schedule {schedule_i+1} | until reaching {current_schedule[1]}, increment: {current_schedule[2]}, step_size: {current_schedule[3]}"
                )
                up = True if args.chunksize < current_schedule[1] else False
                args.increment = current_schedule[2]
                args.step_size = current_schedule[3]
                schedule_complete = False

            if cond() and not schedule_complete:
                args.chunksize = current_schedule[1]
                schedule_complete = True
                args.increment = 0
                args.step_size = 1
                print(
                    f"Exiting schedule {schedule_i+1} | chunksize now: {args.chunksize}, increment: {args.increment}, step_size: {args.step_size}"
                )
                schedule_i += 1
                if schedule_i == len(args.schedule):
                    if args.back_to_default:
                        args.chunksize = base_chunksize
                        args.increment = base_increment
                        args.step_size = base_step_size
                        print(
                            f"Schedules completed | back to chunksize: {args.chunksize}, increment: {args.increment}, step_size: {current_schedule[3]}"
                        )
                    else:
                        print(f"All schedules completed")
                else:
                    current_schedule = args.schedule[schedule_i]

        avg_img, min_img, max_img = initialize_arrays()
        subset = all_images[i : i + args.chunksize]
        for j, img_filename in enumerate(subset):
            print(f"{i}/{N} ({j+1}/{len(subset)})\r", end="")
            current_img = np.array(
                Image.open(os.path.join(args.dir, img_filename)).convert("RGB"),
                dtype=np.uint8,
            )
            avg_img += current_img
            min_img = np.where(current_img <= min_img, current_img, min_img)
            max_img = np.where(current_img >= max_img, current_img, max_img)

        avg_img = avg_img.astype(np.float32) / args.chunksize  # averaging
        # Round values in array and cast as 8-bit integer
        avg_img = np.array(np.round(avg_img), dtype=np.uint8)

        save_bulk(
            (
                [avg_img, f"average.{i}.tiff"],
                [min_img, f"min.{i}.tiff"],
                [max_img, f"max.{i}.tiff"],
            )
        )
        print()
        i += args.step_size
        if args.schedule:
            if up:
                args.chunksize += args.increment
            else:
                args.chunksize -= args.increment
        else:
            args.chunksize += args.increment


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""""")

    parser.add_argument(
        "--dir", "-d", type=str, default="data", help="""The data directory"""
    )

    parser.add_argument(
        "--chunksize",
        "-c",
        type=int,
        default=90,
        help="""
        The chunk size on which the sliding calculation will be performed.
        Default: 90.
        """,
    )

    parser.add_argument(
        "--increment",
        "-n",
        type=int,
        default=0,
        help="""
        The increment by which the chunksize is increased at each step.
        Default: 0.
        """,
    )

    parser.add_argument(
        "--step_size",
        "-p",
        type=int,
        default=1,
        help="""
        The amount of frame to advance by at each iteration.
        Defaults: 1.
        """,
    )

    parser.add_argument(
        "--schedule",
        "-e",
        type=str,
        help="""
        Schedules for more fine-grained changes to chunksizes.
        Format:
            'start_frame,chunksize_goal,increment,step_size'

        These can be chained together with a semicolon:

            's_f,c_g,n,s_s;s_f_2,c_g_2,n_2,s_s_1;s_f_3,c_g_3,n_3,s_s_3;...'

        Example:
            python process.py -e '145,90,1,1;294,45,1,1'
            "From frame 145 onward, start incrementing the chunksize until
            reaching 90. From frame 294, start decrementing until reaching 45.

        Incrementing or decrementing will be deduced given the current
        chunksize at the time the schedule starts.

        The --chunksize, --increment and --step_size  options will still apply
        before the beginning of the first schedule, and, if --back_to_default
        is enabled, after the end of the last one.
        """,
    )

    parser.add_argument(
        "--back_to_default",
        action="store_true",
        help="""
        When using a schedule, go back to the base --chunksize and --increment
        after all schedules are complete. Defaults to false.
        """,
    )

    args = parser.parse_args()

    main(args)
