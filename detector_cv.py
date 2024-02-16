import asyncio
import numpy as np
import cv2
import aiohttp
import time

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

cap = cv2.VideoCapture(0)


async def send_request(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(await response.text())


async def detect_people():

    has_detected = False

    while True:

        state = 0
        current_state = 0
        start_time = time.time()

        ret, frame = cap.read()

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for (xA, yA, xB, yB) in boxes:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        if len(boxes) > 0 and has_detected == False:
            state = 1
            has_detected = True
            if current_state != state:
                print("Found {} people".format(len(boxes)))
                print("Moving forward")
                # send a request to the server
                asyncio.ensure_future(send_request(
                    'http://192.168.4.1/moving'))
                current_state = state

        elif len(boxes) == 0 and has_detected == True:
            state = 0
            has_detected = False
            if current_state != state:
                print("No people found anymore, stopping the robot and crying")
                # send a request to the server
                asyncio.ensure_future(send_request(
                    'http://192.168.4.1/stop_moving'))
                await asyncio.sleep(1)
                asyncio.ensure_future(send_request(
                    'http://192.168.4.1/crying'))
                await asyncio.sleep(6)
                asyncio.ensure_future(send_request(
                    'http://192.168.4.1/stop_crying'))
                current_state = state

        elapsed_time = time.time() - start_time

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(0)


async def main():
    await asyncio.gather(
        detect_people(),
    )

if __name__ == '__main__':
    asyncio.run(main())

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
