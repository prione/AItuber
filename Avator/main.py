import cv2
import torch
import pyvirtualcam
import numpy as np
from PIL import Image
import tha2.poser.modes.mode_20_wx
from models import TalkingAnimeLight, TalkingAnime3
from utils import preprocessing_image
import os
import queue
import time
import math
import random
from collections import OrderedDict
from multiprocessing import Value, Process, Queue
from pyanime4k import ac
from tha2.mocap.ifacialmocap_constants import *
from args import args
from tha3.util import torch_linear_to_srgb
import collections
import glob

def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)


class FPS:
    def __init__(self, avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        if len(self.frametimestamps) > 1:
            return len(self.frametimestamps) / (self.frametimestamps[-1] - self.frametimestamps[0])
        else:
            return 0.0

device = torch.device('cuda') if torch.cuda.is_available() and not args.skip_model else torch.device('cpu')

ifm_converter = tha2.poser.modes.mode_20_wx.IFacialMocapPoseConverter20()


class ModelClientProcess(Process):
    def __init__(self, input_image):
        super().__init__()
        self.should_terminate = Value('b', False)
        self.updated = Value('b', False)
        self.data = None
        self.input_image = input_image
        self.output_queue = Queue()
        self.input_queue = Queue()
        self.model_fps_number = Value('f', 0.0)
        self.gpu_fps_number = Value('f', 0.0)
        self.cache_hit_ratio = Value('f', 0.0)
        self.gpu_cache_hit_ratio = Value('f', 0.0)

    def run(self):
        model = None
        if not args.skip_model:
            model = TalkingAnime3().to(device)
            model = model.eval()
            model = model
            print("Pretrained Model Loaded")

        eyebrow_vector = torch.empty(1, 12, dtype=torch.half if args.model.endswith('half') else torch.float)
        mouth_eye_vector = torch.empty(1, 27, dtype=torch.half if args.model.endswith('half') else torch.float)
        pose_vector = torch.empty(1, 6, dtype=torch.half if args.model.endswith('half') else torch.float)

        input_image = self.input_image.to(device)
        eyebrow_vector = eyebrow_vector.to(device)
        mouth_eye_vector = mouth_eye_vector.to(device)
        pose_vector = pose_vector.to(device)

        model_cache = OrderedDict()
        tot = 0
        hit = 0
        hit_in_a_row = 0
        model_fps = FPS()
        gpu_fps = FPS()
        while True:
            model_input = None
            try:
                while not self.input_queue.empty():
                    model_input = self.input_queue.get_nowait()
            except queue.Empty:
                continue
            if model_input is None: continue
            simplify_arr = [1000] * ifm_converter.pose_size
            if args.simplify >= 1:
                simplify_arr = [200] * ifm_converter.pose_size
                simplify_arr[ifm_converter.eye_wink_left_index] = 50
                simplify_arr[ifm_converter.eye_wink_right_index] = 50
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 50
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 50
                simplify_arr[ifm_converter.eye_surprised_left_index] = 30
                simplify_arr[ifm_converter.eye_surprised_right_index] = 30
                simplify_arr[ifm_converter.iris_rotation_x_index] = 25
                simplify_arr[ifm_converter.iris_rotation_y_index] = 25
                simplify_arr[ifm_converter.eye_raised_lower_eyelid_left_index] = 10
                simplify_arr[ifm_converter.eye_raised_lower_eyelid_right_index] = 10
                simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 5
                simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 5
                simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 5
                simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 5
            if args.simplify >= 2:
                simplify_arr[ifm_converter.head_x_index] = 100
                simplify_arr[ifm_converter.head_y_index] = 100
                simplify_arr[ifm_converter.eye_surprised_left_index] = 10
                simplify_arr[ifm_converter.eye_surprised_right_index] = 10
                model_input[ifm_converter.eye_wink_left_index] += model_input[
                    ifm_converter.eye_happy_wink_left_index]
                model_input[ifm_converter.eye_happy_wink_left_index] = model_input[
                                                                           ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_left_index] = model_input[
                                                                     ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_right_index] += model_input[
                    ifm_converter.eye_happy_wink_right_index]
                model_input[ifm_converter.eye_happy_wink_right_index] = model_input[
                                                                            ifm_converter.eye_wink_right_index] / 2
                model_input[ifm_converter.eye_wink_right_index] = model_input[
                                                                      ifm_converter.eye_wink_right_index] / 2

                uosum = model_input[ifm_converter.mouth_uuu_index] + \
                        model_input[ifm_converter.mouth_ooo_index]
                model_input[ifm_converter.mouth_ooo_index] = uosum
                model_input[ifm_converter.mouth_uuu_index] = 0
                is_open = (model_input[ifm_converter.mouth_aaa_index] + model_input[
                    ifm_converter.mouth_iii_index] + uosum) > 0
                model_input[ifm_converter.mouth_lowered_corner_left_index] = 0
                model_input[ifm_converter.mouth_lowered_corner_right_index] = 0
                model_input[ifm_converter.mouth_raised_corner_left_index] = 0.5 if is_open else 0
                model_input[ifm_converter.mouth_raised_corner_right_index] = 0.5 if is_open else 0
                simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 0
                simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 0
                simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 0
                simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 0
            if args.simplify >= 3:
                simplify_arr[ifm_converter.iris_rotation_x_index] = 20
                simplify_arr[ifm_converter.iris_rotation_y_index] = 20
                simplify_arr[ifm_converter.eye_wink_left_index] = 32
                simplify_arr[ifm_converter.eye_wink_right_index] = 32
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 32
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 32
            if args.simplify >= 4:
                simplify_arr[ifm_converter.head_x_index] = 50
                simplify_arr[ifm_converter.head_y_index] = 50
                simplify_arr[ifm_converter.neck_z_index] = 100
                model_input[ifm_converter.eye_raised_lower_eyelid_left_index] = 0
                model_input[ifm_converter.eye_raised_lower_eyelid_right_index] = 0
                simplify_arr[ifm_converter.iris_rotation_x_index] = 10
                simplify_arr[ifm_converter.iris_rotation_y_index] = 10
                simplify_arr[ifm_converter.eye_wink_left_index] = 24
                simplify_arr[ifm_converter.eye_wink_right_index] = 24
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 24
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 24
                simplify_arr[ifm_converter.eye_surprised_left_index] = 8
                simplify_arr[ifm_converter.eye_surprised_right_index] = 8
                model_input[ifm_converter.eye_wink_left_index] += model_input[
                    ifm_converter.eye_wink_right_index]
                model_input[ifm_converter.eye_wink_right_index] = model_input[
                                                                      ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_left_index] = model_input[
                                                                     ifm_converter.eye_wink_left_index] / 2

                model_input[ifm_converter.eye_surprised_left_index] += model_input[
                    ifm_converter.eye_surprised_right_index]
                model_input[ifm_converter.eye_surprised_right_index] = model_input[
                                                                           ifm_converter.eye_surprised_left_index] / 2
                model_input[ifm_converter.eye_surprised_left_index] = model_input[
                                                                          ifm_converter.eye_surprised_left_index] / 2

                model_input[ifm_converter.eye_happy_wink_left_index] += model_input[
                    ifm_converter.eye_happy_wink_right_index]
                model_input[ifm_converter.eye_happy_wink_right_index] = model_input[
                                                                            ifm_converter.eye_happy_wink_left_index] / 2
                model_input[ifm_converter.eye_happy_wink_left_index] = model_input[
                                                                           ifm_converter.eye_happy_wink_left_index] / 2
                model_input[ifm_converter.mouth_aaa_index] = min(
                    model_input[ifm_converter.mouth_aaa_index] +
                    model_input[ifm_converter.mouth_ooo_index] / 2 +
                    model_input[ifm_converter.mouth_iii_index] / 2 +
                    model_input[ifm_converter.mouth_uuu_index] / 2, 1
                )
                model_input[ifm_converter.mouth_ooo_index] = 0
                model_input[ifm_converter.mouth_iii_index] = 0
                model_input[ifm_converter.mouth_uuu_index] = 0
            for i in range(4, args.simplify):
                simplify_arr = [max(math.ceil(x * 0.8), 5) for x in simplify_arr]
            for i in range(0, len(simplify_arr)):
                if simplify_arr[i] > 0:
                    model_input[i] = round(model_input[i] * simplify_arr[i]) / simplify_arr[i]
            input_hash = hash(tuple(model_input))
            cached = model_cache.get(input_hash)
            tot += 1
            eyebrow_vector_c = [0.0] * 12
            mouth_eye_vector_c = [0.0] * 27
            if cached is not None and hit_in_a_row < self.model_fps_number.value:
                self.output_queue.put_nowait(cached)
                model_cache.move_to_end(input_hash)
                hit += 1
                hit_in_a_row += 1
            else:
                hit_in_a_row = 0
                if args.perf == 'model':
                    tic = time.perf_counter()
                if args.eyebrow:
                    for i in range(12):
                        eyebrow_vector[0, i] = model_input[i]
                        eyebrow_vector_c[i] = model_input[i]
                for i in range(27):
                    mouth_eye_vector[0, i] = model_input[i + 12]
                    mouth_eye_vector_c[i] = model_input[i + 12]
                for i in range(6):
                    pose_vector[0, i] = model_input[i + 27 + 12]
                if model is None:
                    output_image = input_image
                else:
                    output_image = model(input_image, mouth_eye_vector, pose_vector, eyebrow_vector, mouth_eye_vector_c,
                                         eyebrow_vector_c,
                                         self.gpu_cache_hit_ratio)
                if args.perf == 'model':
                    torch.cuda.synchronize()
                    print("model", (time.perf_counter() - tic) * 1000)
                    tic = time.perf_counter()
                postprocessed_image = output_image[0].float()
                if args.perf == 'model':
                    print("cpu()", (time.perf_counter() - tic) * 1000)
                    tic = time.perf_counter()
                postprocessed_image = convert_linear_to_srgb((postprocessed_image + 1.0) / 2.0)
                c, h, w = postprocessed_image.shape
                postprocessed_image = 255.0 * torch.transpose(postprocessed_image.reshape(c, h * w), 0, 1).reshape(h, w,
                                                                                                                   c)
                postprocessed_image = postprocessed_image.byte().detach().cpu().numpy()
                if args.perf == 'model':
                    print("postprocess", (time.perf_counter() - tic) * 1000)
                    tic = time.perf_counter()

                self.output_queue.put_nowait(postprocessed_image)
                if args.debug:
                    self.gpu_fps_number.value = gpu_fps()
                if args.max_cache_len > 0:
                    model_cache[input_hash] = postprocessed_image
                    if len(model_cache) > args.max_cache_len:
                        model_cache.popitem(last=False)
            if args.debug:
                self.model_fps_number.value = model_fps()
                self.cache_hit_ratio.value = hit / tot


@torch.no_grad()
def main():
    img = Image.open(f"{os.path.dirname(__file__)}/data/images/{args.character}.png")
    img = img.convert('RGBA')
    IMG_WIDTH = 512
    wRatio = img.size[0] / IMG_WIDTH
    img = img.resize((IMG_WIDTH, int(img.size[1] / wRatio)))
    for i, px in enumerate(img.getdata()):
        if px[3] <= 0:
            y = i // IMG_WIDTH
            x = i % IMG_WIDTH
            img.putpixel((x, y), (0, 0, 0, 0))
    input_image = preprocessing_image(img.crop((0, 0, IMG_WIDTH, IMG_WIDTH)))
    if args.model.endswith('half'):
        input_image = torch.from_numpy(input_image).half() * 2.0 - 1
    else:
        input_image = torch.from_numpy(input_image).float() * 2.0 - 1

    input_image = input_image.unsqueeze(0)
    extra_image = None
    if img.size[1] > IMG_WIDTH:
        extra_image = np.array(img.crop((0, IMG_WIDTH, img.size[0], img.size[1])))

    print("Character Image Loaded:", args.character)

    output_fps = FPS()

    if args.output_webcam:
        cam_scale = 1
        cam_width_scale = 1
        if args.anime4k:
            cam_scale = 2
        if args.alpha_split:
            cam_width_scale = 2
        cam = pyvirtualcam.Camera(width=args.output_w * cam_scale * cam_width_scale, height=args.output_h * cam_scale,
                                  fps=60,
                                  backend=args.output_webcam,
                                  fmt=
                                  {'unitycapture': pyvirtualcam.PixelFormat.RGBA, 'obs': pyvirtualcam.PixelFormat.RGB}[
                                      args.output_webcam])
        print(f'Using virtual camera: {cam.device}')

    a = None

    if args.anime4k:
        parameters = ac.Parameters()
        # enable HDN for ACNet
        parameters.HDN = True

        a = ac.AC(
            managerList=ac.ManagerList([ac.OpenCLACNetManager(pID=0, dID=0)]),
            type=ac.ProcessorType.OpenCL_ACNet,
        )
        a.set_arguments(parameters)
        print("Anime4K Loaded")

    position_vector = [0, 0, 0, 1]

    model_output = None
    model_process = ModelClientProcess(input_image)
    model_process.daemon = True
    model_process.start()

    print("Ready. Close this console to exit.")


    import sys
    sys.path.append("..")
    import Language


    # motion_generator = Language.chatbot.bot(path, do_memorory=False, do_save=False)

    blink = False
    blink_frame = 0
    temp_path = f"{os.getcwd()}/Temp"

    while True:

        if args.perf == 'main':
            tic = time.perf_counter()
            
        eyebrow_vector_c = [0.0] * 12
        mouth_eye_vector_c = [0.0] * 27
        pose_vector_c = [0.0] * 6

        # 瞬き
        if blink == False:
            if random.random() < 0.002:
                blink = True

        if blink == True:
            mouth_eye_vector_c[2] = math.sin(2*math.pi * blink_frame/100) # 左まぶた
            mouth_eye_vector_c[3] = math.sin(2*math.pi * blink_frame/100) # 右 
            blink_frame += 1

            if blink_frame == 100:
                blink_frame = 0
                blink = False

        eyebrow_vector_c[6]=math.sin(time.perf_counter() * 1.1) # 左まゆ
        eyebrow_vector_c[7]=math.sin(time.perf_counter() * 1.1) # 右まゆ

        if glob.glob(f"{temp_path}/*.wav"):
            mouth_eye_vector_c[14] = math.sin(time.perf_counter() * 13) # 口
        else: 
            mouth_eye_vector_c[14] = 0

        if os.path.isfile(f"{temp_path}/thinking"):
            pose_vector_c[0] = math.sin(time.perf_counter() * 1.5) * 0.5
            pose_vector_c[1] = 1
            pose_vector_c[2] = math.sin(time.perf_counter() * 0.5) * 0.2 
            mouth_eye_vector_c[25] = math.sin(time.perf_counter() * 1.5) * 0.2
            mouth_eye_vector_c[26] = 0.85

        else:   
            pose_vector_c[0] = 0 #顔_上下
            pose_vector_c[1] = 1 * math.sin(time.perf_counter() * 1) # 顔_左右
            pose_vector_c[2] = 1 * math.sin(time.perf_counter() * 0.6) * 0.5 # 首
            mouth_eye_vector_c[25] = 0 # 目_上下
            mouth_eye_vector_c[26] = 1 * math.sin(time.perf_counter() * 1) * 0.8 # 目_左右

        pose_vector_c[3] = pose_vector_c[1] # 体の回転
        pose_vector_c[4] = pose_vector_c[2] # 体のゆれ

        model_input_arr = eyebrow_vector_c
        model_input_arr.extend(mouth_eye_vector_c)
        model_input_arr.extend(pose_vector_c)

        model_process.input_queue.put_nowait(model_input_arr)

        has_model_output = 0
        try:
            new_model_output = model_output
            while not model_process.output_queue.empty():
                has_model_output += 1
                new_model_output = model_process.output_queue.get_nowait()
            model_output = new_model_output
        except queue.Empty:
            pass
        if model_output is None:
            time.sleep(1)
            continue

        postprocessed_image = model_output

        if args.perf == 'main':
            print('===')
            print("input", time.perf_counter() - tic)
            tic = time.perf_counter()

        if extra_image is not None:
            postprocessed_image = cv2.vconcat([postprocessed_image, extra_image])

        k_scale = 1
        rotate_angle = 0
        dx = 0
        dy = 0
        if args.extend_movement:
            k_scale = position_vector[2] * math.sqrt(args.extend_movement) + 1
            rotate_angle = -position_vector[0] * 10 * args.extend_movement
            dx = position_vector[0] * 400 * k_scale * args.extend_movement
            dy = -position_vector[1] * 600 * k_scale * args.extend_movement
        if args.bongo:
            rotate_angle -= 5
        rm = cv2.getRotationMatrix2D((IMG_WIDTH / 2, IMG_WIDTH / 2), rotate_angle, k_scale)
        rm[0, 2] += dx + args.output_w / 2 - IMG_WIDTH / 2
        rm[1, 2] += dy + args.output_h / 2 - IMG_WIDTH / 2

        postprocessed_image = cv2.warpAffine(
            postprocessed_image,
            rm,
            (args.output_w, args.output_h))

        if args.perf == 'main':
            print("extendmovement", (time.perf_counter() - tic) * 1000)
            tic = time.perf_counter()

        output_fps_number = output_fps()

        if args.anime4k:
            alpha_channel = postprocessed_image[:, :, 3]
            alpha_channel = cv2.resize(alpha_channel, None, fx=2, fy=2)

            img1 = cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2BGR)
            # a.load_image_from_numpy(img, input_type=ac.AC_INPUT_BGR)
            a.load_image_from_numpy(img1, input_type=ac.AC_INPUT_BGR)
            a.process()
            postprocessed_image = a.save_image_to_numpy()
            postprocessed_image = cv2.merge((postprocessed_image, alpha_channel))
            postprocessed_image = cv2.cvtColor(postprocessed_image, cv2.COLOR_BGRA2RGBA)
            if args.perf == 'main':
                print("anime4k", (time.perf_counter() - tic) * 1000)
                tic = time.perf_counter()
        if args.alpha_split:
            alpha_image = cv2.merge(
                [postprocessed_image[:, :, 3], postprocessed_image[:, :, 3], postprocessed_image[:, :, 3]])
            alpha_image = cv2.cvtColor(alpha_image, cv2.COLOR_RGB2RGBA)
            postprocessed_image = cv2.hconcat([postprocessed_image, alpha_image])

        if args.debug:
            output_frame = cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2BGRA)
            cv2.putText(output_frame, str('OUT_FPS:%.1f' % output_fps_number), (0, 16), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 255, 0), 1)
            if args.max_cache_len > 0:
                cv2.putText(output_frame, str(
                    'GPU_FPS:%.1f / %.1f' % (model_process.model_fps_number.value, model_process.gpu_fps_number.value)),
                            (0, 32),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            else:
                cv2.putText(output_frame, str(
                    'GPU_FPS:%.1f' % (model_process.model_fps_number.value)),
                            (0, 32),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            if args.max_cache_len > 0:
                cv2.putText(output_frame, str('MEMCACHED:%.1f%%' % (model_process.cache_hit_ratio.value * 100)),
                            (0, 64),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            if args.max_gpu_cache_len > 0:
                cv2.putText(output_frame, str('GPUCACHED:%.1f%%' % (model_process.gpu_cache_hit_ratio.value * 100)),
                            (0, 80),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.imshow("frame", output_frame)
            # cv2.imshow("camera", debug_image)
            cv2.waitKey(1)
        if args.output_webcam:
            # result_image = np.zeros([720, 1280, 3], dtype=np.uint8)
            # result_image[720 - 512:, 1280 // 2 - 256:1280 // 2 + 256] = cv2.resize(
            #     cv2.cvtColor(postprocessing_image(output_image.cpu()), cv2.COLOR_RGBA2RGB), (512, 512))
            result_image = postprocessed_image
            if args.output_webcam == 'obs':
                result_image = cv2.cvtColor(result_image, cv2.COLOR_RGBA2RGB)
            cam.send(result_image)
            cam.sleep_until_next_frame()
        if args.perf == 'main':
            print("output", (time.perf_counter() - tic) * 1000)
            tic = time.perf_counter()


def run():
    import tkinter as tk

    root = tk.Tk()
    root.resizable(False, False)
    main()

if __name__ == "__main__":
    run()