import os
import json
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
from time import sleep

def read_info(info_path):
    with open(info_path) as file:
        info = json.load(file)
    return info
    
def write_info(info_path, info):
    json_object = json.dumps(info, indent = 2)
    with open(info_path, 'w') as file:
        file.write(json_object)
        file.close()

def choose_checkpoint(path, file, i, model):
    bef_path = os.path.join(path, f'output_{model}', file, "splatfacto")
    for f in os.listdir(bef_path):
        bef_path = os.path.join(path, f'output_{model}', file, "splatfacto", f)
        break
    long_path = os.path.join(bef_path, "nerfstudio_models")
    temp_path = os.path.join(bef_path, "temp")
    os.makedirs(temp_path, exist_ok=True)
    os.system(f"mv {long_path}/* {temp_path}")
    
    for fil in os.listdir(temp_path):
        iterarion = int(fil.removeprefix('step-').removesuffix('.ckpt'))
        if iterarion == i or iterarion == i-1:
            os.system(f"mv {temp_path}/{fil} {long_path}")

def return_checkpoints(path, file, model):
    bef_path = os.path.join(path, f'output_{model}', file, "splatfacto")
    for f in os.listdir(bef_path):
        bef_path = os.path.join(path, f'output_{model}', file, "splatfacto", f)
        break
    long_path = os.path.join(bef_path, "nerfstudio_models")
    temp_path = os.path.join(bef_path, "temp")
    os.system(f"mv {temp_path}/* {long_path}")
    os.system(f"rm -rf {temp_path}")

def render(path, file, checks, exp_setting, output_path, model):
    for check in checks:
        choose_checkpoint(path, file, check, model)
        os.system(
            'ns-render ' + 
            'dataset ' + 
            f'--load-config {os.path.join(path, f"output_{model}", file, "*", "*", "config.yml")} ' + 
            f'--output-path {os.path.join(path, f"output_{model}")}'
        )
        return_checkpoints(path, file, model)
        get_output_together(path, exp_setting, output_path, check, model)

def get_num_gaussians(path, file, checks, exp_setting, output_path, model):
    for check in checks:
        iterations = round(check / 10000) * 10000
        choose_checkpoint(path, file, check, model)
        os.system(
            'ns-export ' + 
            'gaussian-splat ' + 
            f'--load-config {os.path.join(path, f"output_{model}", file, "*", "*", "config.yml")} ' + 
            f'--output-dir {os.path.join(path, f"output_{model}")}'
        )
        ply_path = os.path.join(path, f'output_{model}', "splat.ply")
        ply = PlyData.read(ply_path)
        num_gaussians = ply['vertex'].count

        info_path = os.path.join(output_path, exp_setting, 'metrics.json')
        info = read_info(info_path)

        info[str(iterations)]['num_gaussians'] = num_gaussians
        write_info(info_path, info)
        return_checkpoints(path, file, model)

def get_output_together(path, exp_setting, output_path, check, model):
    # Create output path
    os.makedirs(os.path.join(output_path, exp_setting), exist_ok=True)
    os.makedirs(os.path.join(output_path, exp_setting, f'images_{check}'), exist_ok=True)
    output_path = os.path.join(output_path, exp_setting, f'images_{check}')

    out_path = os.path.join(path, f'output_{model}', "test")
    orig_path = os.path.join(out_path, "gt-rgb")
    rendered_path = os.path.join(out_path, "rgb")

    for file in os.listdir(orig_path):
        orig_filepath = os.path.join(orig_path, file)
        rendered_filepath = os.path.join(rendered_path, file)

        orig = Image.open(orig_filepath)
        rendered = Image.open(rendered_filepath)

        combined = Image.new("RGB", (orig.width + rendered.width, orig.height + 50), "white")
        combined.paste(orig, (0, 0))
        combined.paste(rendered, (orig.width, 0))

        draw = ImageDraw.Draw(combined)

        font = ImageFont.truetype("DejaVuSans.ttf", 40)

        draw.text((orig.width // 2 - 40, orig.height + 5), "Ground Truth", fill="black", font=font)
        draw.text((orig.width + rendered.width // 2 - 40, orig.height + 5), "Rendered", fill="black", font=font)

        combined.save(os.path.join(output_path, file))

def save_metrics(path, exp_setting, output_path, model):
    os.makedirs(os.path.join(output_path, exp_setting), exist_ok=True)
    output_path = os.path.join(output_path, exp_setting)

    eval_path = os.path.join(path, f'output_{model}', "evaluations")
    info_sum = {}
    for file in os.listdir(eval_path):
        iterations = round(int(file.removeprefix('eval_ckpt_').removesuffix('.json')) / 10000) * 10000
        info = read_info(os.path.join(eval_path, file))
        info_sum[iterations] = {
            'PSNR': info['results']['psnr'],
            'SSIM': info['results']['ssim'],
            'LPIPS': info['results']['lpips'],
            'FPS': info['results']['fps']
        }
    info = read_info(os.path.join(path, 'output_metrics_features.json'))
    info_sum['tempo_train'] = info[model]['tempo_train']
    write_info(os.path.join(output_path, 'metrics.json'), info_sum)
    # os.system(f"mv {os.path.join(path, 'output_metrics_features.json')} '{output_path}'")

def save_training_setting(path, file, exp_setting, output_path, model):
    bef_path = os.path.join(path, f'output_{model}', file, "splatfacto")
    for f in os.listdir(bef_path):
        bef_path = os.path.join(path, f'output_{model}', file, "splatfacto", f)
        break
    config_path = os.path.join(bef_path, "config.yml")
    os.makedirs(os.path.join(output_path, exp_setting), exist_ok=True)
    os.system(f'mv {config_path} "{os.path.join(output_path, exp_setting)}"')

def reset_training(path, model):
    info_path = os.path.join(path, 'info.json')
    if os.path.exists(info_path):
        info = read_info(info_path)
        info[model]['trained'] = False
        info[model]['evaluations'] = False
        write_info(info_path, info)

    if os.path.exists(os.path.join(path, f'output_{model}')):
        os.system(f"rm -rf {os.path.join(path, f'output_{model}')}")

def run_command(path, tipe:str, es: bool, rae: int, re: int, mi: int, model: str, frames_number: int=None, df: int = 4):
    if frames_number is not None:
        if es:
            os.system(f"python3 main.py -{tipe} {path} -cm exhaustive -df {df} -es {es} -ol 0.025 -rae {rae} -re {re} -mi {mi} -m {model} -fn {frames_number}")
        else:
            os.system(f"python3 main.py -{tipe} {path} -cm exhaustive -df {df} -ol 0.025 -rae {rae} -re {re} -mi {mi} -m {model} -fn {frames_number}")
    else:
        if es:
            os.system(f"python3 main.py -{tipe} {path} -cm exhaustive -df {df} -es {es} -ol 0.025 -rae {rae} -re {re} -mi {mi} -m {model}")
        else:
            os.system(f"python3 main.py -{tipe} {path} -cm exhaustive -df {df} -ol 0.025 -rae {rae} -re {re} -mi {mi} -m {model}")

def pipeline(path, tipe:str, es: bool, rae: int, re: int, mi: int, output_path: str, checks: list, model: str, frames_number: int=None, df: int=4):
    exp_setting = f'num-iterations_{mi//1000}k enhanced-splatfacto_{str(es)} reset-alpha-every_{rae} refine-every_{re}'
    for folder in os.listdir(path):
        try:
            folder_path = os.path.join(path, folder)
            reset_training(folder_path, model)
            run_command(path, tipe, es, rae, re, mi, model, frames_number, df)
            render(folder_path, folder, checks, exp_setting, output_path, model)
            save_metrics(folder_path, exp_setting, output_path, model)
            get_num_gaussians(folder_path, folder, checks, exp_setting, output_path, model)
            save_training_setting(folder_path, folder, exp_setting, output_path, model)
        except Exception as e:
            print(f'Error {e}')
        sleep(60.0 * 5.0)

models = ['splatfacto-big', 'splatfacto-mcmc']
enhanceds = [False, True]
reset_alpha_every_num_iterations = 3000
frames_numbers = [None, 500, 500]
tipes = ['id', 'vd', 'vd']
res = [50,100,200,300,400,500,600,700,800,900]
downscale_factor = 4

num_iterations = 30000
checks = [*range(10000, num_iterations, 10000)]
checks.append(num_iterations - 1)

paths = [
    '/workspace/Documentos/teste_densification/casa',
    '/workspace/Documentos/teste_densification/truck',
    '/workspace/Documentos/teste_densification/train'
]
big_output_paths = [
    '/workspace/Documentos/teste_densification/casa_jsons',
    '/workspace/Documentos/teste_densification/truck_jsons',
    '/workspace/Documentos/teste_densification/train_jsons'
]

for tipe, frames_number, path, big_output_path in zip(tipes, frames_numbers, paths, big_output_paths):
    os.makedirs(big_output_path, exist_ok=True)
    
    for model in models:
        for enhanced in enhanceds:
            if model == 'splatfacto-mcmc' and enhanced:
                pass
            else:
                output_path = os.path.join(big_output_path, f'{model} bilateral_{enhanced}') 
                os.makedirs(output_path, exist_ok=True)

                for re in res:
                    rae = reset_alpha_every_num_iterations // re
                    pipeline(path=path, tipe=tipe, es=enhanced, rae=rae, re=re, mi=num_iterations, output_path=output_path, checks=checks, model=model, df=downscale_factor)