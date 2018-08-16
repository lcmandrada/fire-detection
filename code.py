'''
Fire Detection
Luke Clark M. Andrada

fire detection through image processing
    background subtraction > color analysis > source analysis > display analysis
        > shape analysis > variance analysis > blob detection > alarm decision unit

background subtraction
    higher open_kernel      aggressive noise reduction          stricter detection
    lower close_kernel      passive hole-filling                stricter detection

color analysis
    higher div              fire is more red than blue          stricter detection
    higher mul              fire is more red than blue          stricter detection

source analysis
    lower range             fire is more stagnant               stricter detection

display analysis
    higher gamma            blob is more different              stricter detection

shape analysis
    lower omega             fire is more different              stricter detection

variance analysis
    lower weight            intense fire is more ignored        stricter detection
    higher sigma            fire is more diverse                stricter detection

blob detection
    lower cnt               fewer blobs                         stricter detection
    higher pix              fire is bigger                      stricter detection
    higher barrier          aggressive shape analysis           stricter detection

alarm decision unit
    lower den               fewer final alerts                  stricter detection
    higher len              fewer final alerts                  stricter detection
    lower interval          fewer final alert                   stricter detection
'''

from math import ceil
import cv2
import numpy as np

class FireDetection():
    """ class for fire detection """
    def __init__(self):
        # background subtraction
        self.subtractor = cv2.createBackgroundSubtractorMOG2()

        # refactor
        self.factor = 2

        # morphological kernels
        okern = ceil(3 / self.factor)
        ckern = ceil(30 / self.factor)
        self.open_kernel = np.ones((okern, okern), np.uint8)
        self.close_kernel = np.ones((ckern, ckern), np.uint8)

        # fire and intense fire divider
        self.div = 250
        # cb multiplier
        self.mul = 1.3

        # blob count
        self.cnt = 1
        # minimum blob area
        self.pix = 2
        # shape analysis limit
        self.barrier = 2

        # centroid
        self.xcentroid = None
        self.ycentroid = None
        # centroid distance threshold
        self.range = 20

        # displays
        self.last = None
        self.this = None
        # display analysis threshold
        self.gamma = 0.1

        # histograms
        self.prev = None
        self.cur = None
        # shape analysis threshold
        self.omega = 0.6

        # intense fire multiplier
        self.weight = 1.5
        # variance analysis threshold
        self.sigma = 50

        # alarm history
        self.history = []
        # alert ratio
        self.den = 1.7
        # history length
        self.len = 10
        self.detected = 0
        self.count = 0
        # alarm and alert status
        self.alert = False

        # frame counter
        self.frame = 0
        # false detection interval
        self.interval = 10

    def background_subtraction(self, img):
        """ background subtraction through gaussian mixture model """

        # resize image for lighter processing
        img = cv2.resize(img, (ceil(img.shape[1] / self.factor), ceil(img.shape[0] / self.factor)))

        # copy input
        copy = img.copy()
        # extract foreground mask
        copy = self.subtractor.apply(copy)

        # apply morphological operations
        # opening to reduce noise
        opening = cv2.morphologyEx(copy, cv2.MORPH_OPEN, self.open_kernel)
        # closing to fill holes
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.close_kernel)

        # binarize foreground mask
        _, threshold = cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY)
        # extract foreground from copy
        copy = cv2.bitwise_and(img, cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR))

        # return threshold and extracted foreground
        return img, threshold, copy

    def color_analysis(self, img, threshold):
        """ fire color analysis through YCrCb """

        # convert input to YCrCb space
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

        # initialize block
        block = np.zeros(img.shape, dtype=np.uint8)

        # calculate mean of Y
        count = 0
        y_total = 0
        for y_axis in range(0, block.shape[0]):
            for x_axis in range(0, block.shape[1]):
                # skip if mask pixel is black or irrelevant
                if threshold[y_axis, x_axis] == 0:
                    continue
                y_part, _, _ = ycrcb[y_axis, x_axis]
                y_total += y_part
                count += 1
        if count != 0:
            y_mean = y_total / count
        else:
            y_mean = 0

        # iterate each block pixel
        for y_axis in range(0, block.shape[0]):
            for x_axis in range(0, block.shape[1]):
                # skip if mask pixel is black or irrelevant
                if threshold[y_axis, x_axis] == 0:
                    continue

                # extract pixel intensities
                y_part, cr_part, cb_part = ycrcb[y_axis, x_axis]

                # turn on pixel at coordinates if fire
                if y_part > y_mean:
                    if y_part < self.div:
                        if y_part > cb_part * self.mul and cr_part > cb_part * self.mul:
                            block[y_axis, x_axis] = [255, 255, 255]
                    # turn on pixel at coordinates if intense fire
                    elif y_part > self.div and cr_part > cb_part:
                        block[y_axis, x_axis] = [255, 255, 255]

        # extract blocks considered to contain fire
        copy = cv2.bitwise_and(img, block)

        # return copy with blocks
        return copy

    def blob_detection(self, img, imgz, color, threshold):
        """ detect blobs through contours """

        # count frames
        self.frame += 1

        # append false detection every interval
        if self.frame % self.interval == 0:
            self.alarm_decision(False)

        # find blob contours
        _, contours, _ = cv2.findContours(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), 0, 2)

        # copy input
        blobs = img.copy()

        if contours:
            # select blob with largest contour area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.cnt]

            # get boundary attributes
            x_coord, y_coord, width, height = cv2.boundingRect(contours[0])

            # apply source analysis
            if self.source_analysis(x_coord, y_coord, width, height):
                # stop display analysis if final alert breaks barrier
                if self.count < self.barrier:
                    display = self.display_analysis(width, height)
                else:
                    display = True

                # ignore blobs below minimum area
                if display and width > self.pix and height > self.pix:
                    # stop shape analysis if final alert breaks barrier
                    if self.count < self.barrier:
                        shape = self.shape_analysis(imgz, threshold)
                    else:
                        shape = True

                    # start variance analysis if shape analysis is okay
                    if shape:
                        # bind the blob if variance analysis is okay
                        if self.variance_analysis(color[y_coord:y_coord + height,
                                                        x_coord:x_coord + width]):
                            blobs = cv2.rectangle(blobs,
                                                  (x_coord * self.factor, y_coord * self.factor),
                                                  ((x_coord + width) * self.factor,
                                                   (y_coord + height) * self.factor),
                                                  (255, 255, 255),
                                                  2)

                            # send final alert if alarm ratio exceeds
                            if self.alarm_decision(True) and self.count > self.barrier:
                                # report final alert with counted frames
                                print('Fire Alert! {}x at {}f'.format(self.count, self.frame))

                                # bind the blob if final alert
                                blobs = cv2.rectangle(blobs,
                                                      (x_coord * self.factor,
                                                       y_coord * self.factor),
                                                      ((x_coord + width) * self.factor,
                                                       (y_coord + height) * self.factor),
                                                      (0, 0, 255),
                                                      2)
                                # count detected blobs
                                self.detected += 1

        # return copy with blobs
        return blobs

    def source_analysis(self, x_coord, y_coord, width, height):
        """ source_analysis through centroid """

        # reset state
        flag = False

        # calculate current center coordinates
        xcenter = x_coord + width / 2
        ycenter = y_coord + height / 2

        # accept detection if source location is within threshold
        if self.xcentroid is not None:
            if abs(xcenter - self.xcentroid) < self.range:
                if abs(ycenter - self.ycentroid) < self.range:
                    flag = True

        # save centroid
        self.xcentroid = xcenter
        self.ycentroid = ycenter

        return flag

    def display_analysis(self, width, height):
        """ display analysis through width and height ratio """

        # reset state
        flag = False
        self.this = width / height

        # only accept detection if display ratio exceeds threshold
        if self.last is not None:
            if abs(self.last - self.this) > self.gamma:
                flag = True

        # save ratio
        self.last = self.this

        return flag

    def shape_analysis(self, img, threshold):
        """ shape analysis through histogram """

        # reset state
        flag = False

        # calculate histogram of current input
        self.cur = cv2.calcHist([img],
                                [0, 1, 2],
                                threshold,
                                [256, 256, 256],
                                [0, 150, 0, 150, 0, 150])
        self.cur = cv2.normalize(self.cur, None).flatten()

        # compare current and previous histogram
        if self.prev is not None:
            correl = cv2.compareHist(self.cur, self.prev, cv2.HISTCMP_CORREL)

            # input passes if fire shape is different as per omega
            if correl < self.omega:
                flag = True

        # set current as previous
        self.prev = self.cur

        # return analysis status
        return flag

    def variance_analysis(self, img):
        """ variance analysis on Cr """

        # reset state
        total = 0
        flag = False

        # convert input to yCrCb
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        # calculate mean of Cr
        count = 0
        cr_total = 0
        for y_axis in range(0, img.shape[0]):
            for x_axis in range(0, img.shape[1]):
                # skip if mask pixel is black or irrelevant
                if (img[y_axis, x_axis] == [0, 0, 0]).all():
                    continue
                _, cr_part, _ = ycrcb[y_axis, x_axis]
                cr_total += cr_part
                count += 1
        if count != 0:
            cr_mean = cr_total / count
        else:
            cr_mean = 0

        count = 0
        # iterate each input pixel
        for y_axis in range(0, img.shape[0]):
            for x_axis in range(0, img.shape[1]):
                if (img[y_axis, x_axis] == [0, 0, 0]).all():
                    continue

                # extract pixel intensities
                y_part, cr_part, _ = ycrcb[y_axis, x_axis]

                # calculate variance numerator
                if y_part < 220:
                    total += (cr_part - cr_mean) ** 2
                # for intense fire
                else:
                    total += self.weight * ((cr_part - cr_mean) ** 2)

                count += 1

        # calculate variance
        var = total / count

        # input passes if block is diverse as per sigma
        if var > self.sigma:
            flag = True

        # return analysis status
        return flag

    def alarm_decision(self, flag):
        """ send final alert when alarm ratio exceeds """

        # reset state
        alert = False

        # set current index if alarm
        if flag:
            self.history.append(1)
        # clear current index if no alarm
        else:
            self.history.append(0)

        # set final alert if alarm ratio exceeds
        if len(self.history) > self.len / self.den:
            if sum(self.history) > self.len / self.den:
                self.count += 1
                alert = True
            # reset counter if no-alarm ratio exceeds
            else:
                self.count = 0

        # keep history length
        if len(self.history) > self.len:
            self.history = self.history[1:]

        # return final alert status
        return alert

if __name__ == '__main__':
    # fire detection instance
    FIRE = FireDetection()

    # initialize input
    VIDEO = 'vid/red.h264'
    CAPTURE = cv2.VideoCapture(VIDEO)

    # capture input indefinitely
    while True:
        FLAG, FRAME = CAPTURE.read()

        # break if no input
        if not FLAG:
            break

        # backup input
        COPY = FRAME.copy()
        # apply background subtraction
        COPYZ, THRESHOLD, FOREGROUND = FIRE.background_subtraction(COPY)
        # apply color analysis
        COLOR = FIRE.color_analysis(COPYZ, THRESHOLD)
        # apply blob detection
        BLOBS = FIRE.blob_detection(COPY, COPYZ, COLOR, THRESHOLD)

        # cv2.imshow('color', COLOR)
        cv2.imshow('blobs', BLOBS)

        # exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # destruct
    CAPTURE.release()
    cv2.destroyAllWindows()
