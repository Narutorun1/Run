#!/usr/bin/env python3

import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') or arg.startswith('--execution-providers') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import onnxruntime
import tensorflow
import roop.globals
import roop.metadata
import roop.ui as ui
from roop.predictor import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('--headless-run', action='store_true', help='Run in headless mode')
    program.add_argument('-s', '--source', help='select an source image')
    program.add_argument('-t', '--target', help='select an target image or video')
    program.add_argument('-o', '--output', help='select output file or directory')
    program.add_argument('--frame-processors', nargs='+', choices=['face_swapper', 'face_enhancer'], default=['face_swapper'], help='frame processors')
    program.add_argument('--face-swapper-model', choices=['inswapper_128', 'blendswap_256', 'inswapper_128_fp16', 'simswap_256', 'simswap_512_unofficial'], default='inswapper_128', help='face swapper model')
    program.add_argument('--face-enhancer-model', choices=['codeformer', 'gfpgan_1.2', 'gfpgan_1.3', 'gfpgan_1.4', 'gpen_bfr_256', 'gpen_bfr_512', 'restoreformer_plus_plus'], default='gfpgan_1.3', help='face enhancer model')
    program.add_argument('--face-enhancer-blend', type=int, choices=range(0, 101), default=80, help='face enhancer blend percentage')
    program.add_argument('--reference-face-distance', type=float, default=1.2, help='face distance for recognition')
    program.add_argument('--temp-frame-format', choices=['jpg', 'png', 'bmp'], default='jpg', help='image format for frame extraction')
    program.add_argument('--temp-frame-quality', type=int, choices=range(0, 101), default=100, help='image quality for frame extraction')
    program.add_argument('--keep-temp', action='store_true', help='keep temporary frames')
    program.add_argument('--face-selector-mode', choices=['reference', 'one', 'many'], default='many', help='face selection mode')
    program.add_argument('--face-detector-model', choices=['retinaface', 'yunet'], default='retinaface', help='face detector model')
    program.add_argument('--face-detector-size', choices=['160x160', '320x320', '480x480', '512x512', '640x640', '768x768', '960x960', '1024x1024'], default='640x640', help='face detector input size')
    program.add_argument('--face-detector-score', type=float, default=0.8, help='face detector confidence score')
    program.add_argument('--face-analyser-order', choices=['left-right', 'right-left', 'top-bottom', 'bottom-top', 'small-large', 'large-small', 'best-worst', 'worst-best'], default='large-small', help='face analysis order')
    program.add_argument('--execution-providers', nargs='+', choices=['tensorrt', 'cuda', 'cpu'], default=['cuda'], help='execution providers')
    program.add_argument('--execution-thread-count', type=int, choices=range(1, 129), default=8, help='number of execution threads')
    program.add_argument('--execution-queue-count', type=int, choices=range(1, 33), default=1, help='number of execution queues')
    program.add_argument('--video-memory-strategy', choices=['strict', 'moderate', 'tolerant'], default='moderate', help='video memory strategy')
    program.add_argument('--system-memory-limit', type=int, choices=range(0, 129), default=16, help='system memory limit in GB')
    program.add_argument('--output-video-encoder', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'], default='libx264', help='output video encoder')
    program.add_argument('--output-video-quality', type=int, choices=range(0, 101), default=90, help='output video quality')
    program.add_argument('--output-video-preset', choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'], default='fast', help='output video preset')
    program.add_argument('--output-video-resolution', help='output video resolution')
    program.add_argument('--output-video-fps', type=float, help='output video FPS')
    program.add_argument('--skip-audio', action='store_true', help='skip target audio')
    program.add_argument('--log-level', choices=['error', 'warn', 'info', 'debug'], default='info', help='logging level')
    program.add_argument('--face-mask-types', nargs='+', help='face mask types')
    program.add_argument('--face-mask-blur', type=float, default=0.3, help='face mask blur amount')
    program.add_argument('--face-mask-padding', nargs='+', type=int, help='face mask padding')
    program.add_argument('--face-mask-regions', nargs='+', help='face mask regions')
    program.add_argument('--trim-frame-start', type=int, help='start frame for trimming')
    program.add_argument('--trim-frame-end', type=int, help='end frame for trimming')
    program.add_argument('--frame-enhancer-model', choices=['real_esrgan_x2plus', 'real_esrgan_x4plus', 'real_esrnet_x4plus'], help='frame enhancer model')
    program.add_argument('--frame-enhancer-blend', type=int, choices=range(0, 101), default=100, help='frame enhancer blend percentage')
    program.add_argument('--face-debugger-items', nargs='+', help='face debugger items')
    program.add_argument('--ui-layouts', nargs='+', help='UI layouts')
    program.add_argument('--reference-face-position', type=int, default=0, help='position of the reference face')
    program.add_argument('--reference-frame-number', type=int, default=0, help='number of the reference frame')
    program.add_argument('--face-analyser-age', choices=['child', 'teen', 'adult', 'senior'], help='face analyser age filter')
    program.add_argument('--face-analyser-gender', choices=['male', 'female'], help='face analyser gender filter')
    program.add_argument('--keep-fps', action='store_true', help='keep target fps')
    program.add_argument('--keep-frames', action='store_true', help='keep temporary frames')
    program.add_argument('--many-faces', action='store_true', help='process every face')
    program.add_argument('--similar-face-distance', type=float, default=0.85, help='face distance used for recognition')
    program.add_argument('--max-memory', type=int, help='maximum amount of RAM in GB')
    program.add_argument('--execution-provider', nargs='+', choices=['tensorrt', 'cuda', 'cpu'], default=['cpu'], help='execution provider (backward compatibility)')
    program.add_argument('--execution-threads', type=int, default=suggest_execution_threads(), help='number of execution threads (backward compatibility)')
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()

    roop.globals.source_path = args.source
    roop.globals.target_path = args.target
    roop.globals.output_path = normalize_output_path(args.source, args.target, args.output)
    roop.globals.headless = args.headless_run or (args.source and args.target and args.output)
    roop.globals.frame_processors = args.frame_processors
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_frames = args.keep_temp or args.keep_frames
    roop.globals.skip_audio = args.skip_audio
    roop.globals.many_faces = args.face_selector_mode == 'many' or args.many_faces
    roop.globals.reference_face_position = args.reference_face_position
    roop.globals.reference_frame_number = args.reference_frame_number
    roop.globals.similar_face_distance = args.reference_face_distance if args.reference_face_distance is not None else args.similar_face_distance
    roop.globals.temp_frame_format = args.temp_frame_format
    roop.globals.temp_frame_quality = args.temp_frame_quality
    roop.globals.output_video_encoder = args.output_video_encoder
    roop.globals.output_video_quality = args.output_video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_providers or args.execution_provider)
    roop.globals.execution_threads = args.execution_thread_count if args.execution_thread_count is not None else args.execution_threads
    # Store FaceFusion-specific args for future logic
    roop.globals.face_swapper_model = args.face_swapper_model
    roop.globals.face_enhancer_model = args.face_enhancer_model
    roop.globals.face_enhancer_blend = args.face_enhancer_blend
    roop.globals.face_detector_model = args.face_detector_model
    roop.globals.face_detector_size = args.face_detector_size
    roop.globals.face_detector_score = args.face_detector_score
    roop.globals.face_analyser_order = args.face_analyser_order
    roop.globals.execution_queue_count = args.execution_queue_count
    roop.globals.video_memory_strategy = args.video_memory_strategy
    roop.globals.system_memory_limit = args.system_memory_limit
    roop.globals.output_video_resolution = args.output_video_resolution
    roop.globals.output_video_fps = args.output_video_fps
    roop.globals.face_mask_types = args.face_mask_types
    roop.globals.face_mask_blur = args.face_mask_blur
    roop.globals.face_mask_padding = args.face_mask_padding
    roop.globals.face_mask_regions = args.face_mask_regions
    roop.globals.trim_frame_start = args.trim_frame_start
    roop.globals.trim_frame_end = args.trim_frame_end
    roop.globals.frame_enhancer_model = args.frame_enhancer_model
    roop.globals.frame_enhancer_blend = args.frame_enhancer_blend
    roop.globals.face_debugger_items = args.face_debugger_items
    roop.globals.ui_layouts = args.ui_layouts
    roop.globals.face_analyser_age = args.face_analyser_age
    roop.globals.face_analyser_gender = args.face_analyser_gender

def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]

def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]

def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())

def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1

def limit_resources() -> None:
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))

def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True

def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    if not roop.globals.headless:
        ui.update_status(message)

def start() -> None:
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    if has_image_extension(roop.globals.target_path):
        if predict_image(roop.globals.target_path):
            destroy()
        shutil.copy2(roop.globals.target_path, roop.globals.output_path)
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)
            frame_processor.post_process()
        if is_image(roop.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    if predict_video(roop.globals.target_path):
        destroy()
    update_status('Creating temporary resources...')
    create_temp(roop.globals.target_path)
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Extracting frames with {fps} FPS...')
        extract_frames(roop.globals.target_path, fps)
    else:
        update_status('Extracting frames with 30 FPS...')
        extract_frames(roop.globals.target_path)
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        update_status('Frames not found...')
        return
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Creating video with {fps} FPS...')
        create_video(roop.globals.target_path, fps)
    else:
        update_status('Creating video with 30 FPS...')
        create_video(roop.globals.target_path)
    if roop.globals.skip_audio:
        move_temp(roop.globals.target_path, roop.globals.output_path)
        update_status('Skipping audio...')
    else:
        if roop.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(roop.globals.target_path, roop.globals.output_path)
    update_status('Cleaning temporary resources...')
    clean_temp(roop.globals.target_path)
    if is_video(roop.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')

def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    sys.exit()

def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if roop.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()

if __name__ == '__main__':
    run()
