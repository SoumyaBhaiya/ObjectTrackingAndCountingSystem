import cv2
import numpy as np
import random

# creating a general display screen setting
width, height = 800, 400
fps = 30
duration = 10  # seconds
num_frames = fps * duration

#cars speed/positions/width (basically moving rectangles)
num_cars = 5
car_width, car_height = 40, 20
car_speeds = [random.randint(3, 8) for _ in range(num_cars)]
car_positions = [random.randint(0, height//2) + height//4 for _ in range(num_cars)]  # stay on road
car_x = [random.randint(-200, 0) for _ in range(num_cars)]  # start positions

#video writer(cv2)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('cars_simulation.mp4', fourcc, fps, (width, height))

for frame_num in range(num_frames):
    #black background
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    #drawing road (smaller grey rectangles)
    road_top = height//3
    road_bottom = 2*height//3
    cv2.rectangle(frame, (0, road_top), (width, road_bottom), (50, 50, 50), -1)

    #2 lines for lanes
    for x in range(0, width, 80):
        cv2.line(frame, (x, height//2), (x+40, height//2), (200, 200, 200), 2)

    #updating the cars
    for i in range(num_cars):
        car_x[i] += car_speeds[i]
        if car_x[i] > width + 50:  # reset when car goes off screen
            car_x[i] = -random.randint(100, 300)
            car_positions[i] = random.randint(road_top+10, road_bottom-car_height-10)
            car_speeds[i] = random.randint(3, 8)

        #drawing rectanbles, (cars)
        cv2.rectangle(frame,
                      (car_x[i], car_positions[i]),
                      (car_x[i] + car_width, car_positions[i] + car_height),
                      (255, 255, 255), -1)

    #writing the frame.
    out.write(frame)

    #10 sec preview here.
    cv2.imshow("Car Simulation", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
