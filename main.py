from auditor_utils import *
import argparse

class Properties:
    def add_property(self, name, value):
        setattr(self, name, value)

def cria_pastas(path, is_images=False):
    if not is_images:
        for files in os.listdir(path):
            if not os.path.isdir(os.path.join(path, files)):
                os.system(f"mkdir {os.path.join(path, files)[:os.path.join(path, files).rfind('.')]}")
                os.system(f"mv {os.path.join(path, files)} {os.path.join(path, files)[:os.path.join(path, files).rfind('.')]}")
    else:
        for files in os.listdir(path):
            if os.path.isdir(os.path.join(path, files)):
                os.system(f"mv {os.path.join(path, files)} {os.path.join(path, 'images_orig')}")
                os.system(f"mkdir {os.path.join(path, files)}")
                os.system(f"mv {os.path.join(path, 'images_orig')} {os.path.join(path, files, 'images_orig')}")

def get_video_type(path, dir_file):
    allowed_video_types = ['.mp4', '.MOV', '.mov']
    files = os.listdir(str(os.path.join(path, dir_file)))
    for file in files:
        for video_type in allowed_video_types:
            if file.endswith(video_type):
                return video_type
    return None

def render(path, models):
    for file in os.listdir(path):
        for model in models:
            os.system(
                'ns-render interpolate ' +
                f'--load-config {os.path.join(path, file, "output_" + model, file, "*", "*", "config.yml")} ' +
                f'--output-path {os.path.join(path, file, "output_" + model, file + "_" + model + "_1.mp4")} ' + 
                '--downscale-factor 0.25 ' + 
                '--frame-rate 30 ' + 
                '--interpolation-steps 15'
            )

def main(path, models, is_images=False, propert=None):
    for file in os.listdir(path):
        file_type = get_video_type(path, file)
        if not is_images:
            if file_type is None:
                print(f'Video not found in dir {file}')
                continue
        output = pipeline(
            path,
            file,
            file + str(file_type),
            "pilot",
            os.path.join(path, file),
            os.path.join(path, file, "output"),
            models,
            is_images,
            propert=propert
        )
        write_info(os.path.join(path, file, "output_metrics_features.json"), output)

parser = argparse.ArgumentParser(description="Script with argparse options")
# Add arguments
parser.add_argument("-vd", "--videos_dir", type=str, help="Folder with videos. Do not use ./ to refer to the folder. Use the absolute path.", default=None)
parser.add_argument("-id", "--images_dir", type=str, help="Folder with images folders. Do not use ./ to refer to the folder. Use the absolute path.", default=None)
parser.add_argument("-i", "--initialize", type=bool, help="To initialize the videos folder.", default=False)
parser.add_argument("-r", "--render", type=bool, help="Wether to render videos from models.", default=False)
parser.add_argument("-df", "--downscale_factor", type=int, help="Number of downscale to be used", default=None)
parser.add_argument("-fn", "--frames_number", type=int, help="Number of downscale to be used", default=300)
parser.add_argument("-sf", "--split_fraction", type=float, help="Fraction to divide train/eval dataset", default=0.9)
parser.add_argument("-mi", "--max_num_iterations", type=int, help="Maximum number of iterations during training", default=30000)
parser.add_argument("-cvcw", "--colmap_video_changes_window", type=int, help="Approximate size of window to suggest changes in the video, in case of not finding all poses in one camera model", default=15)
parser.add_argument("-cvcv", "--colmap_video_changes_velocity", type=float, help="Velocity of suggested frames to be changed. Must be less than one, in order to make a slower part of the video", default=0.25)
parser.add_argument("-cl", "--colmap_limit", type=int, help="Number of tries for COLMAP to find all the poses", default=3)
parser.add_argument("-oc", "--only_colmap", type=bool, help="Tell to run only until COLMAP", default=False)
parser.add_argument("-of", "--only_ffmpeg", type=bool, help="Tell to run only until ffmpeg extraction", default=False)
parser.add_argument("-ir", "--is_random", type=bool, help="Wether the dataset of image is random and needs to be sorted", default=False)
parser.add_argument("-m", "--models", nargs='*', type=str, help="Models to run", default=['splatfacto'])
parser.add_argument("-da", "--delete_all", type=bool, help="Delete the unsorted dataset and its stuff to run it again", default=False)
parser.add_argument("-es", "--enhanced_splatfacto", type=bool, help="To make splatfacto with more quality", default=False)
parser.add_argument("-cm", "--colmap_matching", type=str, help="Type of match to use in colmap", default="exhaustive")
parser.add_argument("-ct", "--camera_type", type=str, help="Type of camera to use in colmap", default="perspective")
parser.add_argument("-co", "--camera_optimizer", type=bool, help="If use camera optimizer during training", default=False)
parser.add_argument("-ol", "--opacity_learning", type=float, help="Opacity learning rate to use in training", default=0.05)
parser.add_argument("-rae", "--reset_alpha_every", type=int, help="Number of densifications to reset opacity", default=30)
parser.add_argument("-re", "--refine_every", type=int, help="Number of iterations to make densification", default=100)
parser.add_argument("-mg", "--max_gaussians", type=int, help="Maximum number of mcmc gaussians", default=1000000)

# Parse arguments
args = parser.parse_args()


propert = Properties()
propert.add_property('downscale_factor', args.downscale_factor)
propert.add_property('frames_number', args.frames_number)
propert.add_property('split_fraction', args.split_fraction)
propert.add_property('max_num_iterations', args.max_num_iterations)
propert.add_property('colmap_video_changes_window', args.colmap_video_changes_window)
propert.add_property('colmap_video_changes_velocity', args.colmap_video_changes_velocity)
propert.add_property('colmap_limit', args.colmap_limit)
propert.add_property('only_colmap', args.only_colmap)
propert.add_property('only_ffmpeg', args.only_ffmpeg)
propert.add_property('is_random', args.is_random)
propert.add_property('delete_all', args.delete_all)
propert.add_property('enhanced_splatfacto', args.enhanced_splatfacto)
propert.add_property('colmap_matching', args.colmap_matching)
propert.add_property('camera_type', args.camera_type)
propert.add_property('camera_optimizer', args.camera_optimizer)
propert.add_property('opacity_learning', args.opacity_learning)
propert.add_property('reset_alpha_every', args.reset_alpha_every)
propert.add_property('refine_every', args.refine_every)
propert.add_property('max_gaussians', args.max_gaussians)

models = args.models

if args.initialize:
    if args.videos_dir:
        cria_pastas(args.videos_dir)
    elif args.images_dir:
        cria_pastas(args.images_dir, is_images=True)
else:
    if not args.render:
        if args.images_dir:
            main(args.images_dir, models, is_images=True, propert=propert)
        elif args.videos_dir:
            main(args.videos_dir, models, propert=propert)
    else:
        if args.images_dir:
            render(args.images_dir, models)
        elif args.videos_dir:
            render(args.videos_dir, models)
# propert.is_random = False
# propert.only_colmap = True
# main("/home/tafnes/Downloads/teste_iuri", models, is_images=True, propert=propert)
