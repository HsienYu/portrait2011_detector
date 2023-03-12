import asyncio
import cv2
import aiohttp
import numpy as np
import darknetpy.darknet as darknet


def convert2relative(bbox):
    """
    Convert bounding box format from (x1, y1, w, h) to (x1, y1, x2, y2) relative to image size.
    """
    x, y, w, h = bbox
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return (x1, y1, x2, y2)


async def send_request(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(await response.text())


async def detect_people():

    has_detected = False
    state = 0
    current_state = 0

    # Load the YOLOv4 model
    net = darknet.load_net_custom(
        "cfg/yolov4.cfg".encode('utf-8'), "weights/yolov4.weights".encode('utf-8'), 0, 1)
    meta = darknet.load_meta("cfg/coco.data".encode('utf-8'))

    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Run object detection with YOLOv4
        r = darknet.detect_np(net, meta, frame)
        boxes = [convert2relative(bbox) for _, bbox, _ in r]

        for (xA, yA, xB, yB) in boxes:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        if len(boxes) > 0:
            state = 1
            has_detected = True
            if current_state != state:
                print("Found {} people".format(len(boxes)))
                print("Moving forward")
                # send a request to the server
                asyncio.ensure_future(send_request(
                    'https://duckduckgo.com/?t=osx'))
                current_state = state

        elif len(boxes) == 0 and has_detected == True:
            state = 0
            has_detected = False
            if current_state != state:
                print("No people found anymore, stopping the robot and crying")
                # send a request to the server
                asyncio.ensure_future(send_request(
                    'https://duckduckgo.com/'))
                asyncio.ensure_future(send_request(
                    'https://duckduckgo.com/'))
                current_state = state

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(0)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


async def main():
    await asyncio.gather(
        detect_people(),
    )

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    asyncio.run(main())
