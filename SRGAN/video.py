import cv2
from model import SRGAN
from tensorflow.keras.layers import Input
import numpy as np

video_path = "D:/FPT Res Fes/Code ref/SRGAN/test video/LR_video/LR2.mp4"
video = cv2.VideoCapture(video_path)
srgan = SRGAN()
gen = srgan.generator(Input(shape=(None, None, 3)))
gen.load_weights("D:/FPT Res Fes/Code ref/SRGAN/Gen_120.h5")
# nameOfvideo = get_name_video() + '.mp4'
recording = False

#Video information
fps = video.get(cv2.CAP_PROP_FPS)
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) * 4
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 4
nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(nframes)
print(fps)
# Initialzing object for writing video output
# if recording is False:
#     output = cv2.VideoWriter("plate.mp4", cv2.VideoWriter_fourcc(*'mp4v'),fps , (w,h))
# #   torch.cuda.empty_cache()
#     recording = True

for j in range(nframes):
    ret, img0 = video.read()
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img0 = np.expand_dims(np.array(img0), axis=0)
    img0 = img0 / 127.5 - 1
    sr_frame = gen.predict(img0)
    sr_frame = np.asarray((sr_frame + 1) * 127.5, dtype=np.uint8)
    cv2.imwrite("result/license_plate_{}.png".format(nframes), sr_frame[0][:, :, ::-1])
    # print(sr_frame[0][:, :, ::-1])
    # output.write(sr_frame[0][:, :, ::-1])
    # cv2.waitKey(0)
# video.release()
# output.release()
# cv2.destroyAllWindows()
#   yolo -> gan/low -> ocr
# Closes all the frames