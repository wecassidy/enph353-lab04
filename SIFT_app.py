#!/usr/bin/env python3

from __future__ import print_function, division

import sys

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

from wrapper import VideoCapture


def convert_cv_to_pixmap(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width, channel = cv_img.shape
    bytesPerLine = channel * width
    q_img = QtGui.QImage(
        cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888
    )
    return QtGui.QPixmap.fromImage(q_img)


class SIFTApp(QtWidgets.QMainWindow):
    def __init__(self, camera_number):
        super().__init__()
        loadUi("SIFT_app.ui", self)

        self.fps = 10
        self.camera_enabled = False
        self.template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self.camera = VideoCapture(camera_number)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self.fps)

        self.MIN_MATCH_COUNT = 10
        self.sift = cv2.SIFT_create()
        self.template_keypoints = None
        self.template_descriptors = None

    def SLOT_browse_button(self):
        dialogue = QtWidgets.QFileDialog()
        dialogue.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dialogue.exec_():
            self.template_path = dialogue.selectedFiles()[0]
        else:
            return

        self.template = cv2.imread(self.template_path)
        grey = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.template_keypoints, self.template_descriptors = self.sift.detectAndCompute(
            grey, None
        )

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)
        self.template_loaded = True
        print("Loaded template image file: " + self.template_path)

    def SLOT_query_camera(self):
        ret, frame = self.camera.read()

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(grey, None)

        if self.template_loaded:
            FLANN_INDEX_KDTREE = 1
            matcher = cv2.FlannBasedMatcher(
                {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}, {"checks": 50}
            )
            matches = matcher.knnMatch(self.template_descriptors, descriptors, k=2)

            good = [m for m, n in matches if m.distance < 0.7 * n.distance]
            matchesMask = None
            if len(good) > self.MIN_MATCH_COUNT:
                src_pts = np.float32(
                    [self.template_keypoints[m.queryIdx].pt for m in good]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(
                    -1, 1, 2
                )

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                h, w, _ = self.template.shape
                box = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                ).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(box, M)
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            draw_params = dict(
                matchColor=(0, 255, 0),  # draw matches in green color
                singlePointColor=None,
                matchesMask=matchesMask,  # draw only inliers
                flags=2,
            )
            frame = cv2.drawMatches(
                self.template,
                self.template_keypoints,
                frame,
                keypoints,
                good,
                None,
                **draw_params
            )

        pixmap = convert_cv_to_pixmap(frame)
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self.camera_enabled:
            self._timer.stop()
            self.camera_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self.camera_enabled = True
            self.toggle_cam_button.setText("&Disable camera")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SIFTApp(2)
    window.show()
    app.exec_()
    window.camera.release()
