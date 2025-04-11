import os
import argparse
import sys


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser() # TODO: refine it.
    parser.add_argument("path", default="", help="input path to the video")
    args = parser.parse_args()

    # path must end with / to make sure image path is relative
    if args.path[-1] != '/':
        args.path += '/'
        
    # extract images
    videos = [os.path.join(args.path, vname) for vname in os.listdir(args.path) if vname.endswith(".mp4")]
    images_path = os.path.join(args.path, "images/")
    os.makedirs(images_path, exist_ok=True)
    
    for video in videos:
        cam_name = video.split('/')[-1].split('.')[-2]
        do_system(f"ffmpeg -i {video} -start_number 0 {images_path}{cam_name}_%04d.png")
        
    # load data
    # images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(args.path, "images/", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
    # cams = sorted(set([im[7:12] for im in images]))
    
