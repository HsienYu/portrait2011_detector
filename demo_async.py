#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import time
import argparse

import cv2
import numpy as np
import tensorflow as tf
import asyncio
import aiohttp


from detector import Detector


async def send_request(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(await response.text())


async def detect_and_draw(cap, model, score_th, nms_th):
    while True:

        has_detected = False
        state = 0
        current_state = 0

        start_time = time.time()

        # カメラキャプチャ ################################################
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # 推論実施 ########################################################
        bboxes, scores, class_ids = model.inference(frame)

        if len(bboxes) > 0:
            state = 1
            has_detected = True
            if current_state != state:
                print("Found {} people".format(len(bboxes)))
                print("Moving forward")
                # send a request to the server
                asyncio.ensure_future(send_request(
                    'http://192.168.4.1/moving'))
                current_state = state

        elif len(bboxes) == 0 and has_detected == True:
            state = 0
            has_detected = False
            if current_state != state:
                print("No people found anymore, stopping the robot and crying")
                # send a request to the server
                asyncio.ensure_future(send_request(
                    'http://192.168.4.1/stop_moving'))
                print("Moving stop")
                await asyncio.sleep(1)
                asyncio.ensure_future(send_request(
                    'http://192.168.4.1/crying'))
                print("crying")
                await asyncio.sleep(6)
                asyncio.ensure_future(send_request(
                    'http://192.168.4.1/stop_crying'))
                print("stop crying")
                current_state = state

        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            score_th,
            bboxes,
            scores,
            class_ids,
        )

        # 画面反映 #########################################################
        debug_image = cv2.resize(debug_image, (cap_width, cap_height))
        cv2.imshow('Person Detection Demo', debug_image)

        # Wait for a key event and check if the "ESC" key was pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

        await asyncio.sleep(0)

    cap.release()
    cv2.destroyAllWindows()


def draw_debug(
    image,
    elapsed_time,
    score_th,
    bboxes,
    scores,
    class_ids,
):
    debug_image = copy.deepcopy(image)

    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        # バウンディングボックス
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness=2,
        )

        # クラスID、スコア
        score = '%.2f' % score
        text = '%s:%s' % (str(int(class_id)), score)
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )

    # 推論時間
    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image


async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=360)

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='Path to the TensorFlow Lite object detection model.'
    )

    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help='Detection score threshold.'
    )

    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.5,
        help='Non-maximum suppression threshold.'
    )

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device)
    cap_width = args.width
    cap_height = args.height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    model = Detector(args.model, args.score_threshold, args.nms_threshold)

    await detect_and_draw(cap, model, args.score_threshold, args.nms_threshold)

if __name__ == 'main':
    asyncio.run(main())
