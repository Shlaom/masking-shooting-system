from django.shortcuts import render

from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import threading


import numpy as np
import cv2
import copy
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from keras.models import load_model

class FrameProcessor(object):
    def __init__(self):
        self.minimum_confidence = 0.5
        self.minimum_pixel_size = 10

        self.model_path = './model/keras/facenet_keras.h5'
        self.model = load_model(self.model_path)

        self.face_detector = './face_detector/'
        self.prototxt = self.face_detector + 'deploy.prototxt'
        self.weights = self.face_detector + 'res10_300x300_ssd_iter_140000.caffemodel'
        self.net = cv2.dnn.readNet(self.prototxt, self.weights)

        self.margin = 10
        self.batch_size = 1
        self.n_img_per_person = 30
        self.is_interrupted = False
        self.data = {}
        self.le = None
        self.mean_embs = []
        self.mosaic_margin = 30
        #self.masking_img2 = cv2.imread('./imgs/img3.png', cv2.IMREAD_UNCHANGED)
        #self.masking_img = self.masking_img2[:,:,0:3]
        self.masking_img = cv2.imread('./imgs/img.png', cv2.IMREAD_UNCHANGED)
        self.W = None
        self.H = None
        self.threshold = 0.8
        self.imgs = []

        self.video = cv2.VideoCapture('rtmp://52.79.67.16:1935/live/test')
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def _signal_handler(self, signal, frame):
        self.is_interrupted = True

    def prewhiten(self, x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError('Dimension should be 3 or 4')

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0 / np.sqrt(size))
        y = (x - mean) / std_adj
        return y

    def l2_normalize(self, x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output

    def calc_embs(self, imgs, batch_size):
        aligned_images = self.prewhiten(imgs)
        pd = []
        for start in range(0, len(aligned_images), batch_size):
            pd.append(self.model.predict_on_batch(aligned_images[start:start + batch_size]))
        embs = self.l2_normalize(np.concatenate(pd))

        return embs

    def capture_images(self, frame, name='Unknown'):
        detected_faces_locs = self._detect_faces(frame)
        #print(detected_faces_locs)

        if detected_faces_locs == []:
            print('Faces were not detected in frame..')
            return frame

        largest_face = [0, 0, 0, 0, 0]  # ??? ????????? ????????? ????????? ?????? ??? ?????? ???(?????????) ?????? ?????? ?????? ??????
        for i in detected_faces_locs:
            (left, top, right, bottom) = i

            size = (right - left) * (bottom - top)  # ????????? ????????? ?????? ??????
            if largest_face[0] < size:  # ????????? ????????? ?????? ??? ?????? ?????????(????????????)
                largest_face = [size, left, top, right, bottom]  # ??????

        (left, top, right, bottom) = largest_face[1:5]  # ?????? ??? ????????? ?????? ??????

        img = frame[top:bottom, left:right, :]  # cannot warp image with dimensions (0,0,3)????????? ????????? ?????? ?????? ??????
        if img.shape == (0, 0, 3):
            return frame

        img = resize(frame[top:bottom, left:right, :],
                     (160, 160), mode='reflect')  # ????????? ???????????? ?????????
        self.imgs.append(img)
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=2)

        print("Completed!\n")
        return frame

    def train(self):
        labels = []
        embs = []
        names = self.data.keys()
        print("Preparing training datas...")
        for name, imgs in self.data.items():
            embs_ = self.calc_embs(imgs, self.batch_size)
            labels.extend([name] * len(embs_))
            embs.append(embs_)

        #embs = np.concatenate(embs)
        print("Completed!\n")
        print("Training...")
        le = LabelEncoder().fit(labels)
        y = le.transform(labels)
        print(y)
        ##################################################
        mean_embs = []
        for i in range(0, np.shape(embs)[0]):
            sum = [0] * 128
            for j in embs[i]:
                sum += j
            mean_embs.append(sum/np.shape(embs)[1])
        self.mean_embs.append(mean_embs)
        ##################################################
        #clf = SVC(kernel='linear', probability=True).fit(embs, y)

        self.le = le
        #self.clf = clf
        print("Completed!\n")

    def _findEuclideanDistance(self, src, dst):
        if type(src) == list:
            src = np.array(src)

        if type(dst) == list:
            dst = np.array(dst)

        euclidean_distance = src - dst
        euclidean_distance = np.sqrt(np.sum(np.square(euclidean_distance)))
        return euclidean_distance

    def _apply_margin_to_locs(self, loc):
        (left, top, right, bottom) = loc

        margin_left = max(left - self.mosaic_margin, 0)
        margin_top = max(top - self.mosaic_margin, 0)
        margin_right = min(right + self.mosaic_margin, self.W)
        margin_bottom = min(bottom + self.mosaic_margin, self.H)

        return (margin_left, margin_top, margin_right, margin_bottom)

    def _do_mosaic(self, frame, unknown_imgs_locs):
        for loc in unknown_imgs_locs:
            (left, top, right, bottom) = self._apply_margin_to_locs(loc)

            face_img = frame[top:bottom, left:right]  # ????????? ?????? ????????? crop
            face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04)  # ??????
            face_img = cv2.resize(face_img, (right - left, bottom - top), interpolation=cv2.INTER_AREA)  # ??????
            frame[top:bottom, left:right] = face_img  # ????????? ?????? ?????? ???????????? ??????
        return frame

    def _do_blur(self, frame, unknown_imgs_locs):
        for loc in unknown_imgs_locs:
            (left, top, right, bottom) = self._apply_margin_to_locs(loc)

            face_img = frame[top:bottom, left:right]  # ????????? ?????? ????????? crop
            blurred_img = cv2.blur(face_img, (45, 45), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
            frame[top:bottom, left:right] = blurred_img  # ????????? ?????? ?????? ???????????? ??????
        return frame

    def _do_imaging(self, frame, unknown_imgs_locs):
        for loc in unknown_imgs_locs:
            (left, top, right, bottom) = self._apply_margin_to_locs(loc)

            masking_img = cv2.resize(self.masking_img, dsize = (right-left, bottom-top), interpolation = cv2.INTER_LINEAR)
            #mask = np.full_like(masking_img, 255)
            #mixed = cv2.seamlessClone(masking_img, frame[margin_top:margin_bottom, margin_left:margin_right], mask, ((margin_right-margin_left)//2, (margin_bottom-margin_top)//2), cv2.MIXED_CLONE)
            #frame[margin_top:margin_bottom, margin_left:margin_right] = mixed
            frame[top:bottom, left:right] = masking_img
        return frame

    def _apply_masking(self, frame, unknown_imgs_locs, sign):
        if sign == 1:
            return self._do_mosaic(frame, unknown_imgs_locs)
        elif sign == 2:
            return self._do_blur(frame, unknown_imgs_locs)
        elif sign == 3:
            return self._do_imaging(frame, unknown_imgs_locs)

    def _detect_faces(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)  # setInput() : blob ???????????? ??????????????? ???????????? ??????
        detections = self.net.forward()  # forward() : ???????????? ??????(?????? ??????)

        detected_faces_locs = []

        for i in range(0, detections.shape[2]):
            # ?????? ?????? ?????? ??????
            confidence = detections[0, 0, i, 2]

            # ?????? ?????? ????????? ?????? ???????????? ??? ??????
            if confidence > self.minimum_confidence:
                # bounding box ?????? ??????
                box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])  # w,h???????????? ????????? ??????
                (left, top, right, bottom) = box.astype("int")

                (left, top) = (max(0, left), max(0, top))  # ?????? ?????? ??????
                (right, bottom) = (min(self.W - 1, right), min(self.H - 1, bottom))  # ????????? ????????? ??????

                if (right - left < self.minimum_pixel_size) | (bottom - top < self.minimum_pixel_size):
                    continue

                detected_faces_locs.append((left, top, right, bottom))
        return detected_faces_locs

    def _recognize_faces(self, frame, detected_faces_locs):
        unknown_faces_locs = []

        for locs in detected_faces_locs:
            (left, top, right, bottom) = locs

            img = resize(frame[top:bottom, left:right, :], (160, 160), mode='reflect')  # ????????? ???????????? ?????????
            embs = self.calc_embs(img[np.newaxis], 1)

            threshold = 0.8
            # pred = "Unknown"
            for i in range(0, np.shape(self.mean_embs)[0]):
                # dst = DeepFace.dst.findEuclideanDistance(self.mean_embs[i], embs)
                dst = self._findEuclideanDistance(self.mean_embs[i], embs)
                if dst <= threshold:
                    break
                if (i == (np.shape(self.mean_embs)[0] - 1)):
                    unknown_faces_locs.append((left, top, right, bottom))
        return unknown_faces_locs

    def _optimized_recognize_faces(self, frame, detected_faces_locs):
        cpy_mean_embs = copy.deepcopy(self.mean_embs)

        for locs in detected_faces_locs:
            (left, top, right, bottom) = locs

            img = resize(frame[top:bottom, left:right, :], (160, 160), mode='reflect')  # ????????? ???????????? ?????????
            embs = self.calc_embs(img[np.newaxis], 1)

            for mean_embs in cpy_mean_embs: #????????? ?????? ????????? ??????
                #print(mean_embs)
                dst = self._findEuclideanDistance(mean_embs, embs)
                if dst <= self.threshold:    #?????????????????? ?????? ????????? ????????? ?????? ???????????? ??????, ????????? ????????? ???????????? ??????
                    detected_faces_locs.remove(locs)
                    cpy_mean_embs.remove(mean_embs)
            if len(cpy_mean_embs) == 0:
                break
        return detected_faces_locs

    def get_frame(self):
        frame = self.frame      #self.frame??? ??? ????????? ???????????? ?????? ????????? self.frame??? ???????????? ???????????? ?????? ?????? ?????????. why?
        (self.H, self.W) = frame.shape[:2]

        detected_faces_locs = self._detect_faces(frame)
        unknown_faces_locs = self._optimized_recognize_faces(frame, detected_faces_locs)

        if mode == 1:
            frame = self._apply_masking(frame, unknown_faces_locs, sign)

        frame_flip = cv2.flip(frame, 1)  # ???????????? flip
        _, jpeg = cv2.imencode('.jpg', frame_flip)  # jpeg:????????? ??? ?????????

        return jpeg.tobytes()

    def get_frame_for_registration(self):
        frame = self.frame
        (self.H, self.W) = frame.shape[:2]

        detected_faces_locs = self._detect_faces(frame)
        frame_for_registration = self.capture_images(frame, detected_faces_locs)

        frame_flip = cv2.flip(frame_for_registration, 1)
        _, jpeg = cv2.imencode('.jpg', frame_flip)

        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

# ?????? ?????????
def main(request):
    return render(request, 'main.html')

# ?????? ?????? ?????????
def home(request):
    return render(request, 'home.html')

@gzip.gzip_page
def video(request):
    try:
        # ?????? ????????? ???????????? ?????? ????????? ???????????? ??????????????? ????????? ??????????????? ?????? ???????????? ???????????? ??? ???????????? ??????????????? ??????
        # ???, ???????????? ?????? ???????????? jpeg ???????????? ???????????? ??????????????? ??????, ??????????????? ????????? ????????? ?????? ?????? ???????????? ??? ???????????? ????????? ??????
        #frame_proc = FrameProcessor()
        return StreamingHttpResponse(gen(frame_proc), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def registration(request):
    return render(request, 'registration.html')

def face_capture(request):
    try:
        #frame_proc_for_reg = FrameProcessor()
        return StreamingHttpResponse(gen_for_registration(frame_proc), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass

def gen_for_registration(camera):
    camera.imgs = []
    count = 0

    while True:
        count += 1
        frame = camera.get_frame_for_registration()
        if count == 60:
            break
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    if camera.imgs != []:
        camera.data['Registered_Face'] = np.array(camera.imgs)
        camera.train()
    else:
        print('There are no faces for registration..')

def mypage(request):
    return render(request, 'navBar/myPage/myPage.html')

def masking_on(request):
    global mode
    mode = 1
    return render(request, 'home.html')

def masking_off(request):
    global mode
    mode = 0
    return render(request, 'home.html')

def mode_mosaic(request):
    global sign
    sign = 1
    return render(request, 'home.html')

def mode_blur(request):
    global sign
    sign = 2
    return render(request, 'home.html')

def mode_imaging(request):
    global sign
    sign = 3
    return render(request, 'home.html')

mode = 1
sign = 3
frame_proc = FrameProcessor()