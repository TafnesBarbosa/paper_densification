import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os
import struct
import random
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import json
import cv2
from time import time, sleep
import subprocess
import pandas as pd
import io
import math
import sqlite3
import re
from PIL import Image, ImageDraw, ImageFont
import psutil
import pynvml

def print_progress_bar(iteration, total, bar_length=50):
    progress = (iteration / total)
    arrow = '█'
    spaces = ' ' * (bar_length - int(progress * bar_length))
    print(f'\rProgress: [{arrow * int(progress * bar_length)}{spaces}] {progress * 100:.2f}%', end='', flush=True)

def compute_laplacian(images_paths):
    laplacians = []
    for k, image_path in enumerate(images_paths):
        if image_path.endswith('.png'):
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            var = cv2.Laplacian(gray, cv2.CV_64F).var()
            laplacians.append(var)
            if k % 50 == 0:
                print_progress_bar(k, len(images_paths))
    laplacians = np.array(laplacians)
    return laplacians

def select_windows(frames_number, total_number_frames):
    space = np.linspace(0, total_number_frames, frames_number+1, dtype=int)
    return space[0:frames_number], space[1:]

def apaga_frames_com_mais_blur(frames_path, frames_number, laplacians):
    begs, ends = select_windows(frames_number, len(laplacians))
    for beg, end in zip(begs, ends):
        idx = laplacians[beg:end].argmax() + beg
        for j in range(beg, end):
            if j != idx:
                os.system('rm ' + frames_path + '/frame{:05d}.png'.format(j+1))

def preprocess_images(frames_path):
    images_paths = os.listdir(frames_path)
    images_paths = sorted(images_paths)
    images_paths = [os.path.join(frames_path, image_path) for image_path in images_paths]
    return images_paths

def extrai_frames_ffmpeg(parent_path, video_folder, video_path):
    if not os.path.exists(os.path.join(parent_path, video_folder, 'images_orig')):
        os.system('mkdir ' + os.path.join(parent_path, video_folder, 'images_orig'))
        os.system('ffmpeg -v quiet -i ' + os.path.join(parent_path, video_folder, video_path) + ' ' + os.path.join(parent_path, video_folder, 'images_orig') + r'/frame%5d.png')
    else:
        if len(os.listdir(os.path.join(parent_path, video_folder, 'images_orig'))) == 0:
            os.system('ffmpeg -v quiet -i ' + os.path.join(parent_path, video_folder, video_path) + ' ' + os.path.join(parent_path, video_folder, 'images_orig') + r'/frame%5d.png')

def extrai_frames(parent_path, video_folder, video_path, frames_number, info_path):
    info = read_info(info_path)
    if not info["extract"]:
        extrai_frames_ffmpeg(parent_path, video_folder, video_path)
        laplacians = compute_laplacian(preprocess_images(os.path.join(parent_path, video_folder, 'images_orig')))
        info["extract"] = True
    if not info["delete_blurred"]:
        apaga_frames_com_mais_blur(os.path.join(parent_path, video_folder, 'images_orig'), frames_number, laplacians)
        info["delete_blurred"] = True
    if not info["colmap"] and not info["laplacians"]:
        laplacians = compute_laplacian(preprocess_images(os.path.join(parent_path, video_folder, 'images_orig')))
        info["laplacians"] = True
        info["lap_val"] = laplacians.tolist()
    elif not info["laplacians"]:
        laplacians = compute_laplacian(preprocess_images(os.path.join(parent_path, video_folder, 'images')))
        info["laplacians"] = True
        info["lap_val"] = laplacians.tolist()
    else:
        laplacians = info["lap_val"]
    write_info(info_path, info)
    return laplacians

def delete_colmap_partial_data(colmap_output_path):
    os.system("rm -rf " + os.path.join(colmap_output_path, "images"))
    os.system("rm -rf " + os.path.join(colmap_output_path, "images_2"))
    os.system("rm -rf " + os.path.join(colmap_output_path, "images_4"))
    os.system("rm -rf " + os.path.join(colmap_output_path, "images_8"))
    os.system("rm -rf " + os.path.join(colmap_output_path, "transforms.json"))
    os.system("rm -rf " + os.path.join(colmap_output_path, "sparse_pc.ply"))
    os.system("rm -rf " + os.path.join(colmap_output_path, "colmap", "database.db"))

def get_num_images(images_input_path):
    images_sum = 0
    for x in os.listdir(images_input_path):
        if not os.path.isdir(x):
            images_sum += 1
    return images_sum

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary(path_to_model_file, num_images):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        Qs = [None] * num_images
        Ts = [None] * num_images
        image_names = []
        image_ids = []
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            image_ids.append(image_id)
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = image_name.decode("utf-8")
            image_names.append(image_name)
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
            Qs[image_id - 1] = qvec
            Ts[image_id - 1] = tvec
    
    return Qs, Ts, num_reg_images, image_names, image_ids

def return_maximum_size_reconstruction(colmap_output_path, num_images):
    num_reg_images_max = 0
    folder_max = None
    for folder in os.listdir(os.path.join(colmap_output_path, "colmap", "sparse")):
        if folder.isdigit():
            Q, T, num_reg_images, _, _ = read_images_binary(os.path.join(colmap_output_path, "colmap", "sparse", folder, "images.bin"), num_images)
            if num_reg_images > num_reg_images_max:
                num_reg_images_max = num_reg_images
                Qs = Q
                Ts = T
                folder_max = int(folder)
    
    return Qs, Ts, num_reg_images_max, folder_max

def is_wrong(colmap_output_path, images_path):
    diri_colmap = os.path.join(colmap_output_path, 'colmap/sparse')
    if len(os.listdir(diri_colmap)) == 0:
        return True
    
    # Get number of images extracted of the video
    num_images = get_num_images(images_path)

    # Get the quaternions and translation arrays from the sparse model with the most quantity of poses found
    _, _, num_reg_images_max, _ = return_maximum_size_reconstruction(colmap_output_path, num_images)
    if num_reg_images_max == num_images:
        return False
    return True

def delete_dir(path, diri):
    if os.path.exists(os.path.join(path, diri)):
        os.system('rm -rf ' + os.path.join(path, diri))

def delete_colmap_dirs(colmap_output_path):
    delete_dir(colmap_output_path, 'colmap')
    delete_colmap_partial_data(colmap_output_path)

# Function to get GPU usage
def get_gpu_usage(pid):
    try:
        process_psutil = psutil.Process(pid)
        pids = [pid] + [child.pid for child in process_psutil.children(recursive=True)]
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            res = result.stdout.strip()
            gpu_percentage, gpu_power_draw = int(res.split(', ')[0]), float(res.split(', ')[1])
        except subprocess.CalledProcessError as e:
            print(f"Error querying GPU percentage: {e}")
            gpu_percentage, gpu_power_draw = None, None

        device_count = pynvml.nvmlDeviceGetCount()
        gpu_usage = None

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                # This catches compute and graphics processes
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                procs += pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            except pynvml.NVMLError as e:
                print(f"Error querying GPU usage: {e}")
                gpu_usage = None
                continue
            
            gpu_usage = 0
            for proc in procs:
                if proc.pid in pids:
                    mem_used_mib = proc.usedGpuMemory / 1024**2
                    gpu_usage += mem_used_mib

        return gpu_usage, gpu_percentage, gpu_power_draw
    except:
        return None, None, None
    
# Function to get RAM usage
def get_ram_usage(pid):
    try:
        process_psutil = psutil.Process(pid)
        pids = [pid] + [child.pid for child in process_psutil.children(recursive=True)]
        try:
            result = sum(psutil.Process(pid).memory_info().rss for pid in pids)
            ram_usage = result / 1024 / 1024 / 1024 # In Gb

            return ram_usage
        except psutil.NoSuchProcess:
            print("Error querying RAM usage")
            return None
    except:
        return None

def choose_best_camera_model_and_refine_intrinsics(colmap_output_path, frames_parent_path):
    num_images = get_num_images(os.path.join(frames_parent_path, "images_orig"))
    _, _, _, camera_model = return_maximum_size_reconstruction(colmap_output_path, num_images)
    if camera_model != 0:
        path = os.path.join(colmap_output_path, "colmap", "sparse", f"{camera_model}")
        path_0 = os.path.join(colmap_output_path, "colmap", "sparse", "0")
        path_1 = os.path.join(colmap_output_path, "colmap", "sparse", "_1")
        os.system(f"mv {path_0} {path_1}")
        os.system(f"mv {path} {path_0}")
        os.system(f"mv {path_1} {path}")
        os.system(f"colmap bundle_adjuster --input_path {path_0} --output_path {path_0} --BundleAdjustment.refine_principal_point 1 --BundleAdjustment.max_num_iterations 1000")
        os.system(f"ns-process-data images --data {os.path.join(colmap_output_path, 'images_orig')} --output-dir {colmap_output_path} --matching-method exhaustive --skip-colmap --skip-image-processing")
    return camera_model

def run_command(cmd):
    gpu_vram = []
    gpu_perc = []
    ram = []
    gpu_power = []
    process = subprocess.Popen(cmd)

    # Get PIDs of the cmd
    pid = process.pid

    # Monitor GPU usage while the command is running
    try:
        while process.poll() is None:  # Check if process is still running
            gpu_usage, gpu_percentage, gpu_power_draw = get_gpu_usage(pid)
            ram_usage = get_ram_usage(pid)
            if gpu_usage:
                gpu_vram.append(gpu_usage)
            if gpu_percentage:
                gpu_perc.append(gpu_percentage)
            if gpu_power_draw:
                gpu_power.append(gpu_power_draw)
            if ram_usage:
                ram.append(ram_usage)
            sleep(1)  # Adjust the interval as needed
    finally:
        process.wait()  # Ensure the process completes
        gpu_usage, gpu_percentage, gpu_power_draw = get_gpu_usage(pid)
        ram_usage = get_ram_usage(pid)
        if gpu_usage:
            gpu_vram.append(gpu_usage)
        if gpu_percentage:
            gpu_perc.append(gpu_percentage)
        if gpu_power_draw:
            gpu_power.append(gpu_power_draw)
        if ram_usage:
            ram.append(ram_usage)
    if len(gpu_vram) == 0:
        gpu_vram = [0]
    if len(gpu_perc) == 0:
        gpu_perc = [0]
    if len(gpu_power) == 0:
        gpu_power = [0]
    if len(ram) == 0:
        ram = [0]
    return gpu_vram, gpu_perc, gpu_power, ram

def preprocess_data(frames_parent_path, colmap_output_path, colmap_limit, info_path, propert):
    info = read_info(info_path)
    if not info["colmap"]:
        number_iterations = 0
        is_wrong_flag = True
        start = time()
        while is_wrong_flag and number_iterations < colmap_limit:
            delete_colmap_dirs(colmap_output_path)
            cmd = [
                "ns-process-data", "images", 
                "--data", os.path.join(frames_parent_path, "images_orig"), 
                "--output-dir", colmap_output_path, 
                "--matching-method", f"{propert.colmap_matching}",
                "--no-refine-intrinsics",
                "--camera-type", f"{propert.camera_type}",
                "--verbose"
            ]
            gpu_vram, gpu_perc, gpu_power, ram = run_command(cmd)
            is_wrong_flag = is_wrong(colmap_output_path, os.path.join(frames_parent_path, "images_orig"))
            number_iterations += 1
        
        camera_model = choose_best_camera_model_and_refine_intrinsics(colmap_output_path, frames_parent_path)

        end = time()
        sleep(1.0)
        tempo = end - start

        os.system('rm -rf ' + os.path.join(frames_parent_path, 'images_orig'))
        info["colmap"] = True
        info["gpu_colmap_vram"] = gpu_vram
        info["gpu_colmap_perc"] = gpu_perc
        info["gpu_colmap_power"] = gpu_power
        info["ram_colmap"] = ram
        info["tempo_colmap"] = tempo
        info["colmap_tries"] = number_iterations
        info["camera_model"] = camera_model
        info["colmap_wrong"] = is_wrong_flag
    else:
        gpu_vram = info["gpu_colmap_vram"]
        gpu_perc = info["gpu_colmap_perc"]
        try:
            gpu_power = info["gpu_colmap_power"]
        except:
            gpu_power = None
        ram = info["ram_colmap"]
        tempo = info["tempo_colmap"]
        number_iterations = info["colmap_tries"]
        camera_model = info["camera_model"]
        is_wrong_flag = info["colmap_wrong"]
    write_info(info_path, info)
    return tempo, gpu_vram, gpu_perc, gpu_power, ram, number_iterations, camera_model, is_wrong_flag

def nerfstudio_model(colmap_output_path, splatfacto_output_path, info_path, model, propert=None):
    info = read_info(info_path)
    if not info[model]["trained"]:
        start = time()
        if model == "splatfacto-w":
            cmd = [
                "ns-train", "splatfacto-w-light", 
                "--data", colmap_output_path, 
                "--max-num-iterations", "100000", 
                "--viewer.quit-on-train-completion", "True",
                "--steps-per-save", "10000", 
                "--save-only-latest-checkpoint", "False",
                "--output-dir", splatfacto_output_path,
                "--pipeline.model.enable-bg-model", "True",
                "--pipeline.model.bg-num-layers", "3",
                "--pipeline.model.appearance-embed-dim", "48",
                "--pipeline.model.app-num-layers", "3",
                "--pipeline.model.app-layer-width", "256",
                "--pipeline.model.enable-alpha-loss", "True",
                "--pipeline.model.appearance-features-dim", "72",
                "--pipeline.model.enable-robust-mask", "True",
                "--pipeline.model.never-mask-upper", "0.4",
                "--pipeline.model.sh-degree-interval", "2000",
                "--pipeline.model.use-avg-appearance", "False",
                "--pipeline.model.eval-right-half", "True",
                "nerfstudio-data", "--downscale-factor", str(propert.downscale_factor)
            ]
        else:
            cmd = [
                "ns-train", model, 
                "--data", colmap_output_path, 
                "--max-num-iterations", str(propert.max_num_iterations), 
                "--vis", "viewer+tensorboard", 
                "--viewer.quit-on-train-completion", "True",
                "--steps-per-save", "10000", 
                "--save-only-latest-checkpoint", "False",
                "--output-dir", splatfacto_output_path,
                # "--pipeline.model.cull_alpha_thresh", f"{0.005 if propert.enhanced_splatfacto else 0.1}",
                # "--pipeline.model.continue_cull_post_densification", f"{False if propert.enhanced_splatfacto else True}",
                # "--pipeline.model.densify-grad-thresh", f"{0.0005 if propert.enhanced_splatfacto else 0.0008}",
                # "--pipeline.model.densify-grad-thresh", f"{0.0005}",
                "--pipeline.model.max-gs-num", f"2000000",
                "--optimizers.opacities.optimizer.lr", f"{propert.opacity_learning}",
                "--optimizers.means.optimizer.lr", f"{0.00016}",
                "--pipeline.model.refine-every", f"{propert.refine_every}",
                "--pipeline.model.reset-alpha-every", f"{propert.reset_alpha_every}",
                # "--optimizers.bilateral-grid.scheduler.max-steps", f"{int(propert.max_num_iterations)}",
                # "--optimizers.bilateral-grid.optimizer.lr", f"{propert.bilateral_learning}",
                # "--optimizers.scales.scheduler.lr-final", f"{0.005}",
                # "--pipeline.model.grid-shape", f"{32 if propert.enhanced_splatfacto else 16}", f"{32 if propert.enhanced_splatfacto else 16}", f"{16 if propert.enhanced_splatfacto else 8}",
                # "--pipeline.model.densify-size-thresh", f"{0.0001 if propert.enhanced_splatfacto else 0.01}",
                # "--pipeline.model.cull-scale-thresh", f"{0.1 if propert.enhanced_splatfacto else 0.5}",
                # "--pipeline.model.stop-split-at", f"{25000}",
                "--pipeline.model.use-bilateral-grid", f"{True if propert.enhanced_splatfacto else False}",
                "--pipeline.model.use-scale-regularization", f"{True if propert.enhanced_splatfacto else False}",
                "nerfstudio-data", "--downscale-factor", str(propert.downscale_factor),
                "--train-split-fraction", str(propert.split_fraction)
            ]
        gpu_vram, gpu_perc, gpu_power, ram = run_command(cmd)
        end = time()
        sleep(1.0)
        tempo = end - start
        info[model]["trained"] = True
        info[model][f"gpu_train_vram"] = gpu_vram
        info[model][f"gpu_train_perc"] = gpu_perc
        info[model][f"gpu_train_power"] = gpu_power
        info[model][f"ram_train"] = ram
        info[model][f"tempo_train"] = tempo
    else:
        gpu_vram = info[model][f"gpu_train_vram"]
        gpu_perc = info[model][f"gpu_train_perc"]
        try:
            gpu_power = info[model][f"gpu_train_power"]
        except:
            gpu_power = None
        ram = info[model][f"ram_train"]
        tempo = info[model][f"tempo_train"]
    write_info(info_path, info)
    return tempo, gpu_vram, gpu_perc, gpu_power, ram

def nerfstudio_model_evaluations(model_output_path, video_folder, destino_path, model, info_path, project_path, propert=None):
    info = read_info(info_path)
    elems = [*range(10000, propert.max_num_iterations, 10000)]
    elems.append(propert.max_num_iterations-1)
    if not info[model]["evaluations"]:
        psnr = []
        ssim = []
        lpips = []
        fps = []
        for elem in elems:
            os.system('mv ' + os.path.join(model_output_path, video_folder, '*', '*', 'nerfstudio_models', f'step-{elem:09}.ckpt') + ' ' + os.path.join(model_output_path, video_folder, '*', '*'))
            sleep(1)
        
        for elem in elems:
            os.system('mv ' + os.path.join(model_output_path, video_folder, '*', '*', f'step-{elem:09}.ckpt') + ' ' + os.path.join(model_output_path, video_folder, '*', '*', 'nerfstudio_models'))
            sleep(1)
            os.system('mkdir ' + destino_path)
            os.system('ns-eval --load-config ' + os.path.join(model_output_path, video_folder, '*', '*', 'config.yml') + ' --output-path ' + os.path.join(destino_path, f'eval_ckpt_{elem:09}.json'))
            with open(os.path.join(destino_path, f'eval_ckpt_{elem:09}.json')) as file:
                content = json.load(file)
                psnr.append(content['results']['psnr'])
                ssim.append(content['results']['ssim'])
                lpips.append(content['results']['lpips'])
                fps.append(content['results']['fps'])
        info[model]["evaluations"] = True
        info[model]["psnr"] = psnr
        info[model]["ssim"] = ssim
        info[model]["lpips"] = lpips
        info[model]["fps"] = fps
    else:
        psnr = info[model]["psnr"]
        ssim = info[model]["ssim"]
        lpips = info[model]["lpips"]
        fps = info[model]["fps"]
    elem_best = elems[lpips.index(min(lpips))]
    eval_path = os.path.join(destino_path, f'eval_ckpt_{elem_best:09}.json')
    get_eval_images(project_path, propert.split_fraction, eval_path, model)
    write_info(info_path, info)
    return psnr, ssim, lpips, fps

def get_eval_images(project_path, train_split_fraction, eval_path, ns_model, fontsize=15):
    db_path = os.path.join(project_path, 'colmap', 'database.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count descriptors
    cursor.execute("SELECT image_id, rows AS num_descriptors FROM keypoints;")
    descriptors = cursor.fetchall()
    
    conn.close()

    imagesd = []
    for image, row in descriptors:
        imagesd.append(image)
    imagesd = np.array(imagesd)

    idxs = models_images(os.path.join(project_path, 'colmap', 'sparse'), os.path.join(project_path, 'images_8'))
    num_images = len(idxs)
    num_train_images = math.ceil(num_images * train_split_fraction)
    num_eval_images = num_images - num_train_images
    i_all = np.arange(num_images)
    i_train = np.linspace(
        0, num_images - 1, num_train_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
    assert len(i_eval) == num_eval_images
    i_eval = idxs[i_eval]

    metrics_list = get_metrics_list(eval_path)
    folder_report = f'report/{ns_model}'
    os.system(f'mkdir {os.path.join(project_path, folder_report)}')
    if metrics_list is not None:
        psnr = np.array([elem['psnr'] for elem in metrics_list])
        ssim = np.array([elem['ssim'] for elem in metrics_list])
        lpips = np.array([elem['lpips'] for elem in metrics_list])
        
        PSNR_limit = 25
        fig, ax = plt.subplots(figsize=(12,10))
        plt.plot(imagesd[i_eval], psnr, 'b', linewidth=2)
        plt.plot(imagesd[i_eval], psnr, '.b', markersize=8)
        plt.plot([imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05], [PSNR_limit, PSNR_limit], '--k', linewidth=3)
        arrow = FancyArrowPatch((imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, PSNR_limit), (imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, PSNR_limit+0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])), arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
        arrow = FancyArrowPatch((imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05, PSNR_limit), (imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05, PSNR_limit+0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])), arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
        plt.grid(True)
        plt.ylabel('PSNR', fontsize=fontsize)
        plt.xlabel('Images', fontsize=fontsize)
        plt.title('Evaluation PSNR', fontsize=fontsize)
        plt.tick_params(axis='x', labelsize=fontsize)  # Change xticks fontsize
        plt.tick_params(axis='y', labelsize=fontsize)  # Change yticks fontsize
        plt.savefig(os.path.join(project_path, folder_report, f'PSNR_per_image_{train_split_fraction}.png'), bbox_inches='tight')

        SSIM_limit = 0.9
        fig, ax = plt.subplots(figsize=(12,10))
        plt.plot(imagesd[i_eval], ssim, 'r', linewidth=2)
        plt.plot(imagesd[i_eval], ssim, '.r', markersize=8)
        plt.plot([imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05], [SSIM_limit, SSIM_limit], '--k', linewidth=3)
        arrow = FancyArrowPatch((imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, SSIM_limit), (imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, SSIM_limit+0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])), arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
        arrow = FancyArrowPatch((imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05, SSIM_limit), (imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05, SSIM_limit+0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])), arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
        plt.grid(True)
        plt.ylabel('SSIM', fontsize=fontsize)
        plt.xlabel('Images', fontsize=fontsize)
        plt.title('Evaluation SSIM', fontsize=fontsize)
        plt.tick_params(axis='x', labelsize=fontsize)  # Change xticks fontsize
        plt.tick_params(axis='y', labelsize=fontsize)  # Change yticks fontsize
        plt.savefig(os.path.join(project_path, folder_report, f'SSIM_per_image_{train_split_fraction}.png'), bbox_inches='tight')

        LPIPS_limit = 0.15
        fig, ax = plt.subplots(figsize=(12,10))
        plt.plot(imagesd[i_eval], lpips, 'g', linewidth=2)
        plt.plot(imagesd[i_eval], lpips, '.g', markersize=8)
        plt.plot([imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05], [LPIPS_limit, LPIPS_limit], '--k', linewidth=3)
        arrow = FancyArrowPatch((imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, LPIPS_limit), (imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, LPIPS_limit-0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])), arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
        arrow = FancyArrowPatch((imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05, LPIPS_limit), (imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05, LPIPS_limit-0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])), arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
        plt.grid(True)
        plt.ylabel('LPIPS', fontsize=fontsize)
        plt.xlabel('Images', fontsize=fontsize)
        plt.title('Evaluation LPIPS', fontsize=fontsize)
        plt.tick_params(axis='x', labelsize=fontsize)  # Change xticks fontsize
        plt.tick_params(axis='y', labelsize=fontsize)  # Change yticks fontsize
        plt.savefig(os.path.join(project_path, folder_report, f'LPIPS_per_image_{train_split_fraction}.png'), bbox_inches='tight')
    else:
        with open(os.path.join(project_path, folder_report, 'ALERT.txt'), 'w') as filetext:
            filetext.write('Metrics per image not generated, because Nerfstudio code was not changed as shown in the file changes_nerfstudio_metric_per_image.txt')

def models_images(path_to_model_file, path_images):
    image_idss = []
    models = []

    for folder in sorted(os.listdir(path_to_model_file)):
        image_idsd = get_images_with_pose_ids(os.path.join(path_to_model_file, folder, 'images.bin'), path_images)
        image_idss.append(image_idsd)
        models.append(int(folder))
    idx = models.index(0)
    image_idsd = np.array(image_idss[idx])
    return np.where(image_idsd == 1)[0]

def get_metrics_list(eval_path):
    info = read_info(eval_path)
    if 'results_list' in info.keys():
        return info['results_list']
    else:
        return None

def compute_metrics(camera_positions, normals):
    # Percentage of normals looking to inside
    number_normals_to_inside = 0
    for camera_position, normal in zip(camera_positions, normals):
        cos_angle = (camera_position @ normal) / (np.linalg.norm(camera_position) * np.linalg.norm(normal))
        if cos_angle < 0:
            number_normals_to_inside += 1
    percentage_normals_to_inside = number_normals_to_inside / len(camera_positions)

    # Number of views
    thetas, phis = [], []
    for camera_position in camera_positions:
        _, theta, phi = cartesian_to_spherical(camera_position[0], camera_position[1], camera_position[2])
        thetas.append(theta)
        phis.append(phi)

    return percentage_normals_to_inside, thetas, phis

def plot_number_views(project_path, thetas, phis, M=10, N=20, centered=False, plot=True):
    folder_report = 'report'
    os.system(f'mkdir {project_path}/{folder_report}')
    folder_report = 'report/colmap'
    os.system(f'mkdir {project_path}/{folder_report}')
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # Plot rectangles
    ij = np.zeros((2 * M, N))
    for theta, phi in zip(thetas, phis):
        i = np.floor(M * theta / np.pi)
        j = np.floor(N * phi / np.pi)
        rect = patches.Rectangle((i * np.pi / M, j * np.pi / N), np.pi / M, np.pi / N, color='g')
        if plot:
            ax.add_patch(rect)
        ij[int(i) + M][int(j)] = 1
    percentage_angle_views = sum(sum(ij)) / (2 * M * N)
    
    # Plot grids
    if plot:
        for j in range(N+1):
            ax.plot([-np.pi, np.pi], [j * np.pi/N, j * np.pi/N], 'k', linewidth=0.5)
        for i in range(-M, M+1):
            ax.plot([i * np.pi/M, i * np.pi/M], [0, np.pi], 'k', linewidth=0.5)

        # Plot points
        ax.scatter(thetas, phis, marker='.', c='r')

        if not centered:
            ax.set_title(f"{percentage_angle_views*100:.2f}% of view angles used when not centered")
        else:
            ax.set_title(f"{percentage_angle_views*100:.2f}% of view angles used when centered")

        ax.set_xlabel('theta')
        ax.set_ylabel('phi')

        plt.savefig(os.path.join(project_path, 'report', 'colmap', 'number_of views.png'), bbox_inches='tight')
    return percentage_angle_views

def cartesian_to_spherical(x, y, z):
    # Radial distance (r)
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Azimuthal angle (theta)
    theta = np.arctan2(y, x)
    
    # Polar angle (phi)
    phi = np.arccos(z / r)
    
    return r, theta, phi

def return_camera_positions(Qs, Ts, centered=False):
    camera_positions = []
    normals = []
    for q, t in zip(Qs, Ts):
        if q is not None and t is not None:
            rot = R.from_quat(q)
            rot_matrix = rot.as_matrix()
            camera_positions.append(- rot_matrix.T @ t)
            v = rot_matrix.T @ np.array([0,0,1])
            normals.append(v / np.linalg.norm(v) / 2)
    
    # Center of video
    center = np.mean(np.array(camera_positions), axis=0)

    # Basis of the center of mass
    aux = random.sample(normals, len(normals))
    aux1 = np.mean(aux[:len(aux)//2], axis=0)
    aux1 /= np.linalg.norm(aux1)
    aux2 = np.mean(aux[len(aux)//2:], axis=0)
    aux2 /= np.linalg.norm(aux2)
    w = np.cross(aux1, aux2) / (np.linalg.norm(np.cross(aux1, aux2)))
    Rot = np.array([aux1, np.cross(w, aux1), w]).T
    Rotinv = np.linalg.inv(Rot)
    camera_positions_center = [x - center for x in camera_positions]
    camera_positions_center = [Rotinv @ x for x in camera_positions_center]
    normals_center = [Rotinv @ x for x in normals]
    if centered:
        return camera_positions_center, normals_center, [0,0,1], [0,0,0]
    else:
        return camera_positions, normals, w, center

def preprocess_evaluation_main(colmap_output_path, images_path, propert):
    # Get number of images extracted of the video
    try:
        num_images = get_num_images(images_path[0])
    except FileNotFoundError:
        try:
            num_images = get_num_images(images_path[1])
        except FileNotFoundError:
            try:
                num_images = get_num_images(images_path[2])
            except FileNotFoundError:
                num_images = get_num_images(images_path[3])

    # Get the quaternions and translation arrays from the sparse model with the most quantity of poses found
    Qs, Ts, num_reg_images_max, camera_model = return_maximum_size_reconstruction(colmap_output_path, num_images)
    
    # Get the camera positions and orientations
    camera_positions, normals, _, _ = return_camera_positions(Qs, Ts)
    camera_positions_center, normals_center, _, _ = return_camera_positions(Qs, Ts, True)

    # Compute metrics for the trajectory
    normals_inside, thetas, phis = compute_metrics(camera_positions, normals)
    normals_inside_center, thetas_center, phis_center = compute_metrics(camera_positions_center, normals_center)

    # Plot number of views
    percentage_angle_views = plot_number_views(colmap_output_path, thetas, phis, centered=False, plot=True)
    # percentage_angle_views = None
    percentage_angle_views_center = plot_number_views(colmap_output_path, thetas_center, phis_center, centered=True, plot=True)
    # percentage_angle_views_center = None
        
    # plot_matches_metrics(colmap_output_path, propert, num_reg_images_max != num_images)

    return normals_inside, normals_inside_center, percentage_angle_views, percentage_angle_views_center, num_reg_images_max / num_images, camera_model

def plot_matches_metrics(project_path, propert, generate_video=False):
    folder_report = 'report'
    os.system(f'mkdir {project_path}/{folder_report}')
    folder_report = 'report/colmap'
    os.system(f'mkdir {project_path}/{folder_report}')
    path_images = os.path.join(project_path, 'images_8')
    path_to_model_file = os.path.join(project_path, 'colmap', 'sparse')

    # COLMAP DATABASE
    db_path = os.path.join(project_path, 'colmap', 'database.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT image_id, rows AS num_descriptors FROM keypoints;")
    descriptors = cursor.fetchall()
    
    cursor.execute("SELECT pair_id, rows AS num_matches FROM matches;")
    total_matches = cursor.fetchall()

    cursor.execute("SELECT pair_id, config AS num_geometries FROM two_view_geometries;")
    geometries = cursor.fetchall()

    cursor.execute("SELECT image_id, name AS names FROM images;")
    image_names = cursor.fetchall()
    image_namesd = {}
    for id, name in image_names:
        image_namesd[id] = name
    
    conn.close()

    imagesd = []
    num_descriptors = []
    for image, row in descriptors:
        imagesd.append(image)
        num_descriptors.append(row)

    matchesd = np.zeros((len(imagesd),len(imagesd)))
    for pair, row in total_matches:
        im1 = pair // 2147483647
        im2 = pair % 2147483647
        matchesd[im1-1,im2-1] = row
        matchesd[im2-1,im1-1] = row
    
    matches_configd = np.zeros((len(imagesd),len(imagesd)))
    for pair, row in geometries:
        im1 = pair // 2147483647
        im2 = pair % 2147483647
        matches_configd[im1-1,im2-1] = row
        matches_configd[im2-1,im1-1] = row

    # Matches with cameras found
    colors = ['b', 'r', 'g', 'y', 'c', 'm', 'w']
    k = 0
    plt.figure()
    image_idss = []
    models = []
    nones = np.zeros_like(imagesd, dtype=int)
    for folder in sorted(os.listdir(path_to_model_file)):
        image_idsd = get_images_with_pose_ids(os.path.join(path_to_model_file, folder, 'images.bin'), path_images)
        plt.plot(
            imagesd, 
            100-np.count_nonzero(matches_configd==0, axis=0)*100/len(imagesd) * image_idsd, 
            '-', 
            color=colors[k], 
            label= 'model ' + folder, 
            linewidth=2
        )
        k += 1
        aux = image_idsd
        aux[np.isnan(aux)] = 0
        nones = nones | aux.astype(int)
        image_idss.append(image_idsd)
        models.append('model ' + folder)
    nones = nones.astype(float)
    nones[nones==1] = np.nan
    nones[nones==0] = 1
    if True in (nones==1):
        plt.plot(
            imagesd, 
            100-np.count_nonzero(matches_configd==0, axis=0)*100/len(imagesd) * nones, 
            'x', 
            color='k', 
            label= 'no model', 
            linewidth=2
        )
    plt.xlabel('Images')
    plt.ylabel('Percentage')
    plt.title('Percentage of images matches related with some camera type for the models found')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(project_path, folder_report, 'percentage_cameras_matches.png'), bbox_inches='tight')

    # Matches between images
    plt.figure(figsize=(8,8))
    plt.imshow(matchesd**0.25, cmap='hot')  # 'viridis' is the colormap; you can try 'gray', 'hot', etc.
    plt.colorbar()  # Add a colorbar to indicate value ranges
    plt.title(f"Matches between images")
    plt.xlabel('Images')
    plt.ylabel('Images')
    plt.savefig(os.path.join(project_path, folder_report, 'images_matches.png'), bbox_inches='tight')

    # List of images per model
    lists = []
    for model, image_idsd in zip(models, image_idss):
        aux = np.array(imagesd)[image_idsd == 1].astype(int).tolist()
        aux = [image_namesd[key] for key in aux]
        lists.append(aux)
    aux = np.array(imagesd)[nones == 1].astype(int).tolist()
    aux = [image_namesd[key] for key in aux]
    lists.append(aux)
    
    df = pd.DataFrame(lists).T.fillna(' ')
    models.append('no model')
    df.columns = models
    df.to_csv(os.path.join(project_path, folder_report, 'models_with_images.csv'), index=False, sep='\t')
    if generate_video:
        save_changed_video(lists, models, project_path, window=propert.colmap_video_changes_window, changes_velocity=propert.colmap_video_changes_velocity)

def save_changed_video(lists, models, project_path, window=20, changes_velocity=0.5):
    idxs = []
    for row in lists:
        aux = []
        for elem in row:
            aux.append(
                int(re.search(r'frame_(\d+)\.png', elem).group(1))
            )
        idxs.append(aux)
    sets = {}
    for model, row in zip(models, idxs):
        sets[model] = set(row)
    set_colors = {}
    colors = ['b', 'r', 'g', 'y', 'c', 'm', 'w']
    for model, color in zip(models, colors):
        set_colors[model] = color
    
    sets = def_set_to_improve(sets, window)

    # Calculate the union of all sets
    all_elements = sorted(set.union(*sets.values()))
    positions = {val: idx for idx, val in enumerate(all_elements)}

    # Initialize the bar layers
    num_sets = len(sets)
    bar_layers = np.zeros((num_sets, len(all_elements)), dtype=int)

    # Populate the bar layers for each set
    for i, (set_name, elements) in enumerate(sets.items()):
        for elem in elements:
            bar_layers[i, positions[elem]] = 1

    changes = np.sum(bar_layers,axis=0)
    changes[changes==1] = 0
    changes[changes!=0] = 1
    for i, key in enumerate(sets.keys()):
        for j, keyi in enumerate(sets.keys()):
            if key != keyi:
                if sets[key] <= sets[keyi]:
                    changes[np.array(list(sets[key]))-1] = 0
    for k in range(len(changes)):
        if bar_layers[-1,k] == 1:
            changes[k] = 1
    
    make_changed_video(project_path, changes, changes_velocity)

def def_set_to_improve(sets, N):
  # aumenta sets
  biggest = sorted(set.union(*sets.values()))[-1]
  lowest = sorted(set.union(*sets.values()))[0]
  sets_up = {}
  for key in sets.keys():
    values = list(sets[key])
    if values[0] != lowest:
      values_inic = np.arange(values[0]-N,values[0],dtype=int)
    else:
      values_inic = []
    if values[-1] != biggest:
      values_fim = np.arange(values[-1]+1,values[-1]+N+1,dtype=int)
    else:
      values_fim = []
    sets_up[key] = set(np.concatenate((values_inic,values,values_fim)).tolist())

  return sets_up

def make_changed_video(project_path, changes, changes_velocity):
    images_path = os.path.join(project_path, 'images_2')
    os.system(f"mkdir {os.path.join(project_path, 'report', 'colmap', 'images_changed')}")
    if os.path.exists(os.path.join(project_path, 'report', 'colmap', 'video_changed.mp4')):
        os.system(f"rm {os.path.join(project_path, 'report', 'colmap', 'video_changed.mp4')}")
    i = 1
    for k, file_path in enumerate(sorted(os.listdir(images_path))):
        image_path = os.path.join(images_path, file_path)
        if changes[k] == 1:
            image = open_image(image_path)
            image = set_image_more_red(image)
            image = set_red_border_on_image(image)
            image = set_box_on_image(image)
            save_image(image, os.path.join(project_path, 'report', 'colmap', 'images_changed', f'frame_{i:05}.png'))
            for j in range(int(1/changes_velocity-1)):
                i += 1
                os.system(f"cp {os.path.join(project_path, 'report', 'colmap', 'images_changed', f'frame_{i-1:05}.png')} {os.path.join(project_path, 'report', 'colmap', 'images_changed', f'frame_{i:05}.png')}")
        else:
            os.system(f"cp {image_path} {os.path.join(project_path, 'report', 'colmap', 'images_changed', f'frame_{i:05}.png')}")
        i += 1
    os.system(f"ffmpeg -framerate 30 -i {os.path.join(project_path, 'report', 'colmap', 'images_changed', 'frame_%05d.png')} -c:v libx264 -r 30 -pix_fmt yuv420p {os.path.join(project_path, 'report', 'colmap', 'video_changed.mp4')}")
    os.system(f"rm -rf {os.path.join(project_path, 'report', 'colmap', 'images_changed')}")

def set_box_on_image(image):
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    
    # Define the text and its position
    text = "Adicionar mais vistas"

    width, height = image.size
    
    initial_font_size = 20
    box_scale_factor = np.maximum(width, height) // 160 // 3

    position = (np.maximum(width, height) // 100, np.maximum(width, height) // 100)  # Top-left corner of the text box
    
    # Define the font (optional, requires a TTF font file)
    try:
        font = ImageFont.truetype("arial.ttf", initial_font_size)  # Adjust the font size
    except IOError:
        font = ImageFont.load_default()
    
    # Scale the box and text size
    scaled_font_size = initial_font_size * box_scale_factor
    try:
        scaled_font = ImageFont.truetype("arial.ttf", scaled_font_size)
    except IOError:
        scaled_font = font
    
    scaled_text_bbox = draw.textbbox((0, 0), text, font=scaled_font)
    scaled_text_width = scaled_text_bbox[2] - scaled_text_bbox[0]
    scaled_text_height = scaled_text_bbox[3] - scaled_text_bbox[1]
    
    # Define the scaled red box position and size
    padding = 10 * box_scale_factor  # Scale the padding as well
    box_position = position  # Top-left corner of the red box
    box = [
        box_position,
        (box_position[0] + scaled_text_width + 2 * padding, box_position[1] + scaled_text_height + 2 * padding),
    ]

    box_width = scaled_text_width + 2 * padding
    box_height = scaled_text_height + 2 * padding
    
    # Draw the red rectangle (scaled)
    draw.rectangle(box, fill="red")
    
    # Calculate the new text position to center it in the scaled red box
    box_center_x = box_position[0] + box_width // 2
    box_center_y = box_position[1] + box_height // 2
    text_x = box_center_x - scaled_text_width // 2
    text_y = box_center_y - scaled_text_height // 2
    text_position = (text_x, text_y)
    
    # Draw the scaled text inside the red box
    draw.text(text_position, text, fill="white", font=scaled_font)

    return image

def set_image_more_red(image):
    image = image.convert("RGB")

    red_overlay = Image.new("RGB", image.size, (255, 0, 0))

    transparency = 0.22

    red_image = Image.blend(image, red_overlay, transparency)

    return red_image

def set_red_border_on_image(image):
    draw = ImageDraw.Draw(image)

    width, height = image.size
    
    border_thickness = np.maximum(width, height) // 100

    draw.rectangle(
        [(0,0), (width, height)],
        outline="red",
        width=border_thickness
    )

    return image

def open_image(image_path):
    # Open an image
    image = Image.open(image_path)
    return image

def save_image(image, path_to_save):
    # Save or show the image
    image.save(path_to_save)

def get_images_with_pose_ids(path_to_model_file, path_images):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    num_images = get_num_images(path_images)
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        Qs = [None] * num_images
        Ts = [None] * num_images
        image_names = []
        image_ids = np.ones((num_images,)) * np.nan
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            image_ids[image_id-1] = 1
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = image_name.decode("utf-8")
            image_names.append(image_name)
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
    return image_ids

def sort_images_based_on_matches(project_path):
    db_path = os.path.join(project_path, 'colmap', 'database.db')
    _, matches = get_matches(db_path)

    N = 5
    matches_ids = None
    idx_max, _, _ = sort_with_shift(matches, matches_ids, 0, N, end = None)

    _, matches = get_matches(db_path)
    matches_ids = None
    _, _, matches_ids = sort_with_shift(matches, matches_ids, 0, N, end = idx_max)

    return matches_ids

def create_resorted_dataset(project_path, matches_ids, propert):
    path = os.path.join(project_path, 'images')
    path_new = os.path.join(project_path, 'images_resorted')
    os.system(f"mkdir {path_new}")
    for k, id in enumerate(matches_ids):
        os.system(f'cp {os.path.join(path, f"frame_{id+1:05}.png")} {os.path.join(path_new, f"frame_{k+1:05}.png")}')
    if propert.delete_all:
        delete_colmap_dirs(project_path)
        os.system("rm -rf " + os.path.join(project_path, "info.json"))
        os.system(f"mv {path_new} {os.path.join(project_path, 'images_orig')}")
    else:
        os.system(f"ffmpeg -framerate 30 -i {os.path.join(path_new, 'frame_%05d.png')} -c:v libx264 -r 30 -pix_fmt yuv420p {os.path.join(project_path, 'report', 'colmap', 'video_only_resorted.mp4')}")
        os.system(f"rm -rf {path_new}")

def get_matches(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count matches
    cursor.execute("SELECT pair_id, rows AS num_matches FROM matches;")
    total_matches = cursor.fetchall()

    cursor.execute("SELECT image_id, name AS algo FROM images;")
    image_names = cursor.fetchall()
    
    conn.close()

    images = {}
    for image_id, name in image_names:
        images[image_id] = name
        
    matches = np.zeros((len(images.keys()),len(images.keys())))
    for pair, row in total_matches:
        im1 = pair // 2147483647
        im2 = pair % 2147483647
        matches[im1-1,im2-1] = row
        matches[im2-1,im1-1] = row

    return images, matches

def shift_row_col(mat, row_col_be, row_col_af):
    mat11, mat12, mat21, mat22 = select_quadrants(mat, row_col_be)
    mat1 = join_quadrants(mat11, mat12, mat21, mat22)
    mat11, mat12, mat21, mat22 = select_quadrants(mat1, row_col_af,True)
    col = mat[:, row_col_be].tolist()
    poped = col.pop(row_col_be)
    if row_col_af > len(col):
        col.append(poped)
    else:
        col.insert(row_col_af, poped)
    col_to_ins = np.array([col]).T
    row_to_ins = np.array([col])
    
    top = np.hstack((mat11, col_to_ins[:row_col_af], mat12))  # Horizontally stack the top row
    bottom = np.hstack((mat21, col_to_ins[row_col_af+1:], mat22))  # Horizontally stack the bottom row
    result = np.vstack((top, row_to_ins, bottom))  # Vertically stack top and bottom
    # row_be = result[]
    return result

def select_quadrants(mat, idx, with_row_column=False):
    if with_row_column:
        mat11 = mat[:idx,:idx]
        mat21 = mat[idx:,:idx]
        mat12 = mat[:idx,idx:]
        mat22 = mat[idx:,idx:]
    else:
        mat11 = mat[:idx,:idx]
        mat21 = mat[idx+1:,:idx]
        mat12 = mat[:idx,idx+1:]
        mat22 = mat[idx+1:,idx+1:]
    return mat11, mat12, mat21, mat22

def join_quadrants(q11, q12, q21, q22):
    top = np.hstack((q11, q12))  # Horizontally stack the top row
    bottom = np.hstack((q21, q22))  # Horizontally stack the bottom row
    result = np.vstack((top, bottom))  # Vertically stack top and bottom
    return result

def sort_matches(matches, matches_ids, k):
    for i in range(len(matches)-k-1):
        idx_be = get_k_sorted(matches[i,i+k:],0)+i+k
        if idx_be > i+k:
            matches = shift_row_col(matches, idx_be, i+k+1)
            matches_ids = change_ids(matches_ids, i+k+1, idx_be)
    return matches, matches_ids

def change_ids(ids, k, idx_be):
    poped = ids.pop(idx_be)
    ids.insert(k, poped)
    return ids
    
def get_k_sorted(lst, k):
    lst_sorted = sorted(lst.tolist(), reverse=True)
    return lst.tolist().index(lst_sorted[k])

def sort_with_shift(matches_orig, matches_ids_orig, k, N, end = None):
    trace_max = -np.inf
    trace_max_ind = None
    if matches_ids_orig == None:
        matches_ids_orig = [*range(len(matches_orig))]
    if end == None:
        for i in range(len(matches_orig)):
            if i > 0:
                matches, matches_ids = initial_shift(matches_orig, matches_ids_orig, i)
                matches, matches_ids = sort_matches(matches, matches_ids, k)
            else:
                matches, matches_ids = sort_matches(matches_orig.copy(), np.array(matches_ids_orig).tolist(), k)
            trace = measure_diagonality(matches, N)
            if trace > trace_max:
                trace_max = trace
                trace_max_ind = matches_ids_orig[i]
            if i % 10 == 0:
                print_progress_bar(i, len(matches_orig))
    else:
        for i in range(matches_ids_orig.index(end)+1):
            if i > 0:
                matches, matches_ids = initial_shift(matches_orig, matches_ids_orig, i)
                matches, matches_ids = sort_matches(matches, matches_ids, k)
            else:
                matches, matches_ids = sort_matches(matches_orig.copy(), np.array(matches_ids_orig).tolist(), k)
            if i % 10 == 0:
                print_progress_bar(i, matches_ids_orig.index(end)+1)
    return trace_max_ind, matches, matches_ids

def initial_shift(mat, mat_ids, steps):
    matc = mat.copy()
    matc_ids = mat_ids.copy()
    for i in range(steps):
        matc = shift_row_col(matc, 0, len(matc)-1)
        matc_ids = change_ids(matc_ids, len(matc)-1, 0)
    return matc, matc_ids

def measure_diagonality(matches, N):
    trace = np.trace(matches, offset = 0)
    for i in range(1, N+1):
        trace += np.trace(matches, offset=i)
        trace += np.trace(matches, offset=-i)
    return trace

def init(parent_path, video_folder, is_images=False):
    if not os.path.exists(os.path.join(parent_path, video_folder, "info.json")):
        info = {
            "extract": is_images,
            "delete_blurred": is_images,
            "laplacians": False,
            "pilot": False,
            "colmap": False,
            "nerfacto": {
                "trained": False,
                "evaluations": False
            },
            "nerfacto-big": {
                "trained": False,
                "evaluations": False
            },
            "splatfacto": {
                "trained": False,
                "evaluations": False
            },
            "splatfacto-big": {
                "trained": False,
                "evaluations": False
            },
            "splatfacto-w": {
                "trained": False,
                "evaluations": False
            },
            "splatfacto-w-light": {
                "trained": False,
                "evaluations": False
            },
            "splatfacto-mcmc": {
                "trained": False,
                "evaluations": False
            }
        }
        write_info(os.path.join(parent_path, video_folder, "info.json"), info)
    return os.path.join(parent_path, video_folder, "info.json")

def read_info(info_path):
    with open(info_path) as file:
        info = json.load(file)
    return info
    
def write_info(info_path, info):
    json_object = json.dumps(info, indent = 2)
    with open(info_path, 'w') as file:
        file.write(json_object)
        file.close()

def pipeline(parent_path, video_folder, video_path, pilot_output_path, colmap_output_path, splatfacto_output_path, models, is_images=False, propert=None):
    pynvml.nvmlInit()

    # repetition_number = 10
    colmap_limit = propert.colmap_limit
    elems = [*range(10000, propert.max_num_iterations, 10000)]
    elems.append(propert.max_num_iterations-1)

    frames_parent_path = os.path.join(parent_path, video_folder)
    images_path = os.path.join(frames_parent_path, 'images_orig')
    images_path_8 = [
        os.path.join(frames_parent_path, 'images_8'), 
        os.path.join(frames_parent_path, 'images_4'), 
        os.path.join(frames_parent_path, 'images_2'), 
        os.path.join(frames_parent_path, 'images')
    ]

    # Init
    info_path = init(parent_path, video_folder, is_images)

    # Extract frames and get laplacians
    laplacians = extrai_frames(parent_path, video_folder, video_path, propert.frames_number, info_path)

    if propert.only_ffmpeg:
        output = {
                "lap_mean": np.mean(laplacians), 
                "lap_max": max(laplacians), 
                "lap_min": min(laplacians), 
                "lap_median": np.median(laplacians),
            }
    else:

        # Pilot study with repetitions
        # pilot_study(repetition_number, frames_parent_path, pilot_output_path, info_path)

        # Preprocess dataset
        tempo_colmap, gpu_colmap_vram, gpu_colmap_perc, gpu_colmap_power, ram_colmap, number_iterations_colmap, camera_model, is_wrong_flag = preprocess_data(frames_parent_path, colmap_output_path, colmap_limit, info_path, propert)
        
        # Colmap evaluations
        normals_inside, normals_inside_center, percentage_angle_views, percentage_angle_views_center, percentage_poses_found, _ = preprocess_evaluation_main(colmap_output_path, images_path_8, propert)
        
        # Colmap pilot study evaluations
        # normals_inside_pilot, normals_inside_center_pilot, percentage_angle_views_pilot, percentage_angle_views_center_pilot, percentage_poses_found_pilot, camera_models_pilot = colmap_evaluation_pilot(os.path.join(frames_parent_path, pilot_output_path), images_path_8)

        output = {
                "lap_mean": np.mean(laplacians), 
                "lap_max": max(laplacians), 
                "lap_min": min(laplacians), 
                "lap_median": np.median(laplacians),
                
                "tempo_colmap": tempo_colmap, 
                "gpu_colmap_max_vram": max(gpu_colmap_vram), 
                "gpu_colmap_max_perc": max(gpu_colmap_perc), 
                "gpu_colmap_power": sum(gpu_colmap_power) / 1000 / 3600 if gpu_colmap_power is not None else None, # Em kWh
                "ram_colmap_max": max(ram_colmap), 
                "number_iterations_colmap": number_iterations_colmap,

                "percentage_normals_inside": normals_inside, 
                "percentage_normals_inside_center": normals_inside_center, 
                "percentage_angle_views": percentage_angle_views, 
                "percentage_angle_views_center": percentage_angle_views_center, 
                "percentage_poses_found": percentage_poses_found,
                "camera_model": camera_model,
                "wrong_colmap": is_wrong_flag,

                # "percentage_normals_inside_pilot": normals_inside_pilot,
                # "percentage_normals_inside_center_pilot": normals_inside_center_pilot, 
                # "percentage_angle_views_pilot": percentage_angle_views_pilot, 
                # "percentage_angle_views_center_pilot": percentage_angle_views_center_pilot, 
                # "percentage_poses_found_pilot": percentage_poses_found_pilot,
                # "camera_models_pilot": camera_models_pilot,
            }
        if propert.is_random:
            matches_ids = sort_images_based_on_matches(colmap_output_path)
            output['matches_ids'] = matches_ids
            create_resorted_dataset(colmap_output_path, matches_ids)
        if not propert.only_colmap:
            # Models
            for model in models:
                tempo_train, gpu_train_vram, gpu_train_perc, gpu_train_power, ram_train = nerfstudio_model(colmap_output_path, splatfacto_output_path + f"_{model}", info_path, model, propert=propert)
            
                # Model evaluations
                psnr, ssim, lpips, fps = nerfstudio_model_evaluations(splatfacto_output_path + f"_{model}", video_folder, os.path.join(frames_parent_path, f'output_{model}', 'evaluations'), model, info_path, colmap_output_path, propert=propert)

                output[model] = {}
                
                output[model]["tempo_train"] = tempo_train
                output[model]["gpu_train_max_vram"] = max(gpu_train_vram)
                output[model]["gpu_train_max_perc"] = max(gpu_train_perc)
                output[model]["gpu_train_power"] = sum(gpu_train_power) / 1000 / 3600 if gpu_train_power is not None else None # Em kWh
                output[model]["ram_train_max"] = max(ram_train)

                output[model]["psnr_train_max"] = max(psnr)
                output[model]["ssim_train_max"] = max(ssim)
                output[model]["lpips_train_min"] = min(lpips)
                output[model]["fps_train_min"] = min(fps)

                output[model]["psnr_train_max_ckpt"] = elems[psnr.index(max(psnr))]
                output[model]["ssim_train_max_ckpt"] = elems[ssim.index(max(ssim))]
                output[model]["lpips_train_min_ckpt"] = elems[lpips.index(min(lpips))]
                output[model]["fps_train_min_ckpt"] = elems[fps.index(min(fps))]
    
    return output
