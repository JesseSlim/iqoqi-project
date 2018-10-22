from JsCamera import JsCamera
import matplotlib.pyplot as plt
import time

imglist = list()
#plt.ion()

start_time = time.time()
prev_time = start_time

def show_image(imgarray):
    global start_time, prev_time
    cur_time = time.time()
    time_since_start = cur_time - start_time
    time_since_prev = cur_time - prev_time
    prev_time = cur_time
    print(".. callback called after: %f s since start, %f s since previous" % (time_since_start, time_since_prev))
    print(".. image array size: %s" % str(imgarray.shape))
    #plt.figure()
    #plt.imshow(imgarray)
    #plt.show()

cam = JsCamera()
cam.connect()
# cam.print_summary()

cam.capture_sequence_and_collect_images(1, show_image, shutterspeed=0.004, colormode="RGB")

cam.disconnect()

#while True:
#    plt.pause(1)