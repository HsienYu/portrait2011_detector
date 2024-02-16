import asyncio
import numpy as np
import cv2
import aiohttp
import time

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# cv2.startWindowThread()  # Commented out: Disable window thread

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set lower resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

REQUEST_PEOPLE_URL = 'http://192.168.4.1/moving'
REQUEST_STOP_MOVING_URL = 'http://192.168.4.1/stop_moving'
REQUEST_CRYING_URL = 'http://192.168.4.1/crying'
REQUEST_STOP_CRYING_URL = 'http://192.168.4.1/stop_crying'

# REQUEST_PEOPLE_URL = 'http://google.com'
# REQUEST_STOP_MOVING_URL = 'http://google.com'
# REQUEST_CRYING_URL = 'http://google.com'
# REQUEST_STOP_CRYING_URL = 'http://google.com'

request_status = False
has_request_been_sent = False
has_tears = False
no_person_start_time = None


async def send_request(url):
    global request_status
    global has_request_been_sent
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    print("Request sent successfully")
                    request_status = True
                    has_request_been_sent = False
                else:
                    print("Failed to send request")
                    request_status = False
                    has_request_been_sent = False

    except aiohttp.ClientError as e:
        print("Sending request failed:", str(e))
        request_status = False
        has_request_been_sent = False


async def detect_people():
    global no_person_start_time
    global has_tears
    global request_status
    global has_request_been_sent

    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, (320, 240))  # Resize frame
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        boxes, weights = hog.detectMultiScale(
            frame, winStride=(4, 4))  # Adjust HOG parameters

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for (xA, yA, xB, yB) in boxes:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        if len(boxes) > 0:
            print("Found {} people".format(len(boxes)))
            no_person_start_time = None
            if not has_request_been_sent:
                has_request_been_sent = True
                asyncio.create_task(send_request(REQUEST_PEOPLE_URL))
                print("Request status:", request_status)
                has_tears = False

        elif len(boxes) == 0:
            if no_person_start_time is None:
                no_person_start_time = time.time()  # Start measuring no person time

            elapsed_no_person_time = time.time() - no_person_start_time
            if elapsed_no_person_time >= 10 and not has_tears:
                print("No person detected for 10 seconds")
                # Do something when no person is detected for 10 seconds
                if not has_request_been_sent:
                    asyncio.create_task(send_request(REQUEST_STOP_MOVING_URL))
                    await asyncio.sleep(5)
                    asyncio.create_task(send_request(REQUEST_CRYING_URL))
                    await asyncio.sleep(6)
                    asyncio.create_task(send_request(REQUEST_STOP_CRYING_URL))
                    has_tears = True

        # cv2.imshow('frame', frame)  # Commented out: Disable frame display
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        await asyncio.sleep(0.1)  # Increase frame read delay


async def main():
    await asyncio.gather(
        detect_people(),
    )


if __name__ == '__main__':
    asyncio.run(main())

cap.release()
# cv2.destroyAllWindows()  # Commented out: Disable window destruction
# cv2.waitKey(1)  # Commented out: Disable frame display
