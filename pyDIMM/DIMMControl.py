"""
Python app to control the IQOQI DIMM measurement setup.

Author: Jesse Slim, 2016
"""

import os
import os.path
import sys

import datetime
import time
import queue
from enum import Enum
from io import BytesIO
from multiprocessing import Lock, Process, Queue

import ImageConversion, SpotFitting
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtWidgets, QtGui, QtChart

from JsCamera import JsCamera


# compile the main window QtDesigner ui file into a python file
# I do this so the IDE understands where the Ui_MainWindow class comes from and is able to provide hints
os.system("pyuic5 mainWindow.ui > mainWindow.py")
# if the external compilation doesn't work, the following line loads the python class directly from the ui-file
# but you get no type hints in the IDE
# from PyQt5.uic import loadUiType
# Ui_MainWindow, QMainWindow = loadUiType('mainWindow.ui')

from mainWindow import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow


# This extension of the standard QWidget UI-element is used to display the preview and spot images.
# You can provide it with an image using the update_image() method. Cropping and repositioning the image is also done
# by the widget itself if you provide the 'center' and 'zoom' arguments to the update_image() method
#
# In addition it provides a 'clicked'-event that can be subscribed to, which is used by the program to reposition the
# spot previews.
class PreviewWidget(QtWidgets.QWidget):
    """
    Image preview display widget.
    Extension of the standard QWidget UI element that is used to display the preview and spot images in this program

    You can provide it with an image using the update_image() method. Cropping and repositioning the image is also done
    by the widget itself if you provide the 'center' and 'zoom' arguments to the update_image() method

    In addition it provides a 'clicked'-event that can be subscribed to, which is used by the program to reposition the
    spot previews.
    """

    clicked = QtCore.pyqtSignal(QtGui.QMouseEvent, name='clicked')
    """
    Click signal (in the sense of the pyqt framework)
    """


    def __init__(self, parent=None, size=np.array([450, 300]), full_size=np.array([450, 300])):
        """
        Initialize the PreviewWidget

        Args:
            parent:     parent widget in which this widget is to be displayed
            size:       the size of *this* preview widget
            full_size:  the size of the *overview* preview widget. Used to determine the scale of this image so that it
                        corresponds to the scale of the overview image
        """
        super().__init__()
        self.size = size
        self.full_size = full_size
        self.setParent(parent)
        self.upperleft_ipx = None
        self.lowerright_ipx = None
        self.image = None
        self.img_size = None

    def mouseReleaseEvent(self, event):
        if self.isEnabled():
            self.clicked.emit(event)

    def paintEvent(self, event):
        """
        Override of the paint event method which is called by the Qt framework in every UI draw cycle.
        Here we put in the image rendering code.
        """
        if self.image:
            qp = QtGui.QPainter(self)
            target = QtCore.QRectF(0.0, 0.0, self.size[0], self.size[1])
            q_upperleft = QtCore.QPointF(self.upperleft_ipx[0], self.upperleft_ipx[1])
            q_lowerright = QtCore.QPointF(self.lowerright_ipx[0], self.lowerright_ipx[1])
            source = QtCore.QRectF(q_upperleft, q_lowerright)

            qp.drawImage(target, self.image, source)


    def compute_initial_figure(self):
        pass

    def update_image(self, imgdata, cpopt=ImageConversion.ColorProcessingOptions.cpColor, center=None, zoom=1.0):
        """
        Update the image to be displayed

        Args:
            imgdata:    RGB-array containing the new image to be displayed
            cpopt:      specification of the color processing method to be used
            center:     position to center the image on (in image-pixels)
            zoom:       zoom factor of the image (with respect to the *overview* preview)

        Returns:
            None
        """
        self.img_size = np.array(imgdata.shape[1::-1]) # invert the x and y-axis, images are saved in the opposite column/row-order

        converted_imgdata = ImageConversion.apply_color_processing(imgdata, cpopt)

        self.image = QtGui.QImage(converted_imgdata, self.img_size[0], self.img_size[1], self.img_size[0] * 3, QtGui.QImage.Format_RGB888)

        # if we have no center given, take the midpoint of the image
        if center is None:
            center = self.img_size / 2

        display_pix_per_image_pix = self.full_size / self.img_size
        scale_factor = np.min(display_pix_per_image_pix)

        crop_ipx_size = self.size / (scale_factor * zoom)
        self.upperleft_ipx = center - (crop_ipx_size / 2.0)
        self.lowerright_ipx = center + (crop_ipx_size / 2.0)

        self.update()

    def dpx_to_ipx(self, dpx):
        """
        Convert position in display pixels to position in image pixels

        Args:
            dpx:    [x,y]-position array in display pixels to be converted

        Returns:
            [x,y]-position array in image pixels
        """
        rel_pos = dpx / self.size
        ipx = self.upperleft_ipx + (self.lowerright_ipx - self.upperleft_ipx) * rel_pos
        return ipx

    def get_image_rect(self):
        """
        Return the current crop rectangle of the displayed image in image pixels, taking into account the selected
        center and zoom factor.

        Returns:
            (x1, y1, x2, y2) crop rectangle (in integer image pixel coordinates)

        """
        return (int(self.upperleft_ipx[0]), int(self.upperleft_ipx[1]), int(self.lowerright_ipx[0]), int(self.lowerright_ipx[1]))


class ContinuousCollectionThread(QtCore.QThread):
    """
    Continuous collection thread. When running, this thread continuously issues single trigger commands to the camera
    and retrieves the images afterwards. The next trigger command is only issued after the previous image has been
    retrieved. Upon successful retrieval of an image, the 'collected' signal is emitted with the image data as argument.

    Collection can be stopped by setting the 'stop'-attribute to True
    """

    collected = QtCore.pyqtSignal(tuple, int, int, name="collected")
    """
    Signal emitted when an image has been succesfully retrieved.

    Signal args:
        imgdata:        tuple(numpy_image_matrix, jpeg_buffer) that contains the image data for the received image
        image_number:   number of this image in the sequence (always 1 in this case)
        burstnumber:    length of the burst this image belongs to (always 1 in this case)
    """

    def __init__(self, camera_object, camera_lock, parent=None):
        super().__init__()
        self.camera_object = camera_object
        self.camera_lock = camera_lock
        self.stop = False
        """Boolean signal variable to stop the collection thread"""

    def run(self):
        """Continuous collection thread main method"""
        self.camera_lock.acquire()
        while not self.stop:
            time.sleep(0.01)
            self.camera_object.set_burstnumber(1)
            self.camera_object.trigger_capture()
            res = None
            while True:
                time.sleep(0.01)
                res = self.camera_object.check_and_collect_image(1, "RGB")
                if res:
                    break
            self.collected.emit(res, 1, 1)
        self.camera_lock.release()


class BurstCollectionThread(QtCore.QThread):
    """
    Burst collection thread. When running, this thread issues a single burst capture command. After the burst has been
    completed, all captured images are collected one by one. Upon successful retrieval of an image, the 'collected'
    signal is emitted with the image data as argument. After there are no more images available for retrieval, the
    'finished' signal is emitted and the thread exits.
    """

    collected = QtCore.pyqtSignal(tuple, int, int, name="collected")
    """
    Signal emitted when an image has been successfully retrieved.

    Signal args:
        imgdata:        tuple(numpy_image_matrix, jpeg_buffer) that contains the image data for the received image
        image_number:   number of this image in the sequence
        burstnumber:    length of the burst this image belongs to (always 1 in this case)
    """

    finished = QtCore.pyqtSignal(int, bool, name="finished")
    """
    Signal emitted when all images have been retrieved.

    Signal args:
        images_received:        total number of images received
        all_images_received:    boolean to indicate whether the number of images received matches the number of images
                                expected.
    """

    FIRST_COLLECTION_TIMEOUT = 30   # wait a maximum of 30s for the first image to arrive (we may have to wait for the
                                    # burst to finish
    COLLECTION_TIMEOUT = 5          # wait a maximum of 5s for subsequent images to arrive

    def __init__(self, camera_object, camera_lock, burstnumber, parent=None):
        super().__init__()
        self.camera_object = camera_object
        self.camera_lock = camera_lock
        self.burstnumber = burstnumber

    def run(self):
        self.camera_lock.acquire()
        self.camera_object.set_burstnumber(self.burstnumber)
        # triggering burst capture
        self.camera_object.trigger_capture()

        time_since_prev_image = time.time()
        image_number = 0

        # keep collecting images until we time out
        while True:
            time.sleep(0.01)
            res = self.camera_object.check_and_collect_image(1, "RGB")
            if res:
                image_number += 1
                time_since_prev_image = time.time()
                self.collected.emit(res, image_number, self.burstnumber)

            time_passed = time.time() - time_since_prev_image
            if (image_number == 0 and time_passed > self.FIRST_COLLECTION_TIMEOUT) or (
                    image_number > 0 and time_passed > self.COLLECTION_TIMEOUT):
                # timeout! break out
                break

        self.camera_lock.release()
        self.finished.emit(image_number, image_number == self.burstnumber)


class FitDataThread(QtCore.QThread):
    """
    Fit data thread. This thread is used to fit data in the background, so that the UI isn't blocked. Each instance of
    the thread fits two spots in a single image and terminates afterwards. The image data has to be supplied upon
    construction of the thread object. The results are communicated through the 'fitted' signal.

    If the thread is started with the 'killable' argument set to True, the thread may be killed by setting the 'kill'
    attribute to True. Otherwise the thread will just continue execution until it is finished.
    """

    fitted = QtCore.pyqtSignal(tuple, SpotFitting.FittingMethods, name="fitted")
    """
    Signal emitted when the fitting process has finished

    Signal args:
        fitparams:      tuple(fitparamsA, fitparamsB) of fitted parameters for both spots. See the data fitting module
                        for details on the fit parameters
        fitting_method: fitting method that was applied
    """

    def __init__(self, imgarray, spotA_rect, spotB_rect, cpopt, fitting_method, killable = True):
        super().__init__()
        # copy the image array so this thread has it's own copy and doesn't get upset if the image is changed
        self.imgarray = np.copy(imgarray)
        self.spotA_rect = spotA_rect
        self.spotB_rect = spotB_rect
        self.cpopt = cpopt
        self.fitting_method = fitting_method
        self.killable = killable
        self.kill = False
        self.queue = Queue()
        self.subprocess = Process(target=self.subprocess_run, args=(self.imgarray, self.spotA_rect, self.spotB_rect,
                                                                    self.cpopt, self.fitting_method, self.queue))

    def subprocess_run(self, imgarray, spotA_rect, spotB_rect, cpopt, fitting_method, q=None):

        converted_imgdata = ImageConversion.apply_color_processing(imgarray, cpopt)

        spotA_imgarray, iy_A, ix_A, _, _ = ImageConversion.crop_image(converted_imgdata, spotA_rect)
        spotB_imgarray, iy_B, ix_B, _, _ = ImageConversion.crop_image(converted_imgdata, spotB_rect)

        fitparams = SpotFitting.fit_spots_from_array(spot_imgarrays=(spotA_imgarray, spotB_imgarray),
                                                     method=self.fitting_method,
                                                     crop_offsets=(np.array([ix_A, iy_A]), np.array([ix_B, iy_B])))
        if q:
            q.put(fitparams)
        else:
            return fitparams

    def run(self):
        if self.killable:
            # if the fitting process should be killable, we launch it as a subprocess (which may be forcibly killed)
            # communication with the subprocess is done using a Queue
            self.subprocess.start()
            while not self.kill:
                time.sleep(0.01)
                try:
                    fitparams = self.queue.get(False)
                    self.fitted.emit(fitparams, self.fitting_method)
                    return
                except queue.Empty:
                    pass

            # we have been killed!
            self.subprocess.terminate()
            self.queue = None
        else:
            # if the fitting process doesn't have to be killable, just launch it directly inside this thread
            fitparams = self.subprocess_run(self.imgarray, self.spotA_rect, self.spotB_rect, self.cpopt, self.fitting_method)
            self.fitted.emit(fitparams, self.fitting_method)
            return


class Main(QMainWindow, Ui_MainWindow):
    """
    Main UI class that defines the main window and contains code for handling all UI actions (camera communication upon
    button clicks etc.). All code within this class is run inside the GUI thread, so code that takes a long time to
    execute will block all UI interactions and updates. For this reason some tasks have been delegated to separate
    threads (see the above class definitions).

    The graphical layout of the main window is contained in the Ui_MainWindow base class, which is generated from the
    Qt Designer file 'mainWindow.ui'
    """
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # initialize class attributes that will be used later on
        self.camera_object = None
        self.camera_lock = Lock()
        self.cont_shooting_on = False
        self.burst_shooting_on = False
        self.connected = False

        self.continuousCollectionThread = None
        self.burstCollectionThread = None
        self.fitDataThread = None

        self.output_dir = None
        self.output_prefix = None
        self.save_results = None
        self.save_crops_only = None
        self.burstnumber = None
        self.results_file = None
        self.settings_file = None
        self.cpopt = None

        # set up some initial image
        b = np.array([0, 0, 0])
        r = np.array([255, 0, 0])

        self.current_image = np.array([
            [b, b, b, b, b, b, b, b, b],
            [b, b, b, r, r, b, b, b, b],
            [b, b, b, r, r, r, b, b, b],
            [b, b, b, b, r, r, r, b, b],
            [b, b, b, b, b, r, r, b, b],
            [b, b, b, b, b, b, b, b, b]
        ], dtype=np.uint8)

        self.previewSize = np.array([450, 300])
        self.spotSize = np.array([150, 150])

        self.spotAPosition = None
        self.spotBPosition = None

        # define the zoom values corresponding to each slider position
        self.zoom_values = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])

        for fitopt in SpotFitting.FittingMethods:
            self.fittingCombo.addItem(fitopt.name)

        self.fittingCombo.setCurrentIndex(0)

        # set up the preview widgets (can't be done in Qt Designer because the widgets are non-standard)
        self.cameraPreviewPlot = PreviewWidget(parent=self.cameraPreview, size=self.previewSize, full_size=self.previewSize)
        self.spotAPreviewPlot = PreviewWidget(parent=self.spotAPreview, size=self.spotSize, full_size=self.previewSize)
        self.spotBPreviewPlot = PreviewWidget(parent=self.spotBPreview, size=self.spotSize, full_size=self.previewSize)

        # set up the intensity plots
        # the plots themselves will be filled later on.
        self.spotAIntensityChartView = QtChart.QChartView()
        self.spotAIntensityChartView.setParent(self.spotAIntensity)
        self.spotAIntensityChartView.setFixedSize(self.spotAIntensity.size())
        self.spotAIntensityChartView.setRenderHint(QtGui.QPainter.Antialiasing)

        self.spotBIntensityChartView = QtChart.QChartView(parent=self.spotBIntensity)
        self.spotBIntensityChartView.setParent(self.spotBIntensity)
        self.spotBIntensityChartView.setFixedSize(self.spotBIntensity.size())
        self.spotBIntensityChartView.setRenderHint(QtGui.QPainter.Antialiasing)

        # update all dynamic widgets
        self.update_all_data_displays()

        # connect ui events to the corresponding handling methods
        self.connectButton.clicked.connect(self.connect_clicked)
        self.contShootButton.clicked.connect(self.cont_shooting_clicked)
        self.browseOutputDirButton.clicked.connect(self.browse_output_dir_clicked)
        self.startBurstButton.clicked.connect(self.start_burst_clicked)
        self.saveImgButton.clicked.connect(self.save_current_image)

        self.shutterSpeedCombo.activated.connect(self.shutterspeed_changed)
        self.ISOCombo.activated.connect(self.iso_changed)
        self.fittingCombo.activated.connect(self.start_data_fitting)

        self.spotAZoomSlider.valueChanged.connect(self.update_zooms)
        self.spotBZoomSlider.valueChanged.connect(self.update_zooms)

        self.cameraPreviewPlot.clicked.connect(self.preview_click)
        self.spotAPreviewPlot.clicked.connect(self.preview_click)
        self.spotBPreviewPlot.clicked.connect(self.preview_click)

        self.colorProcessingGroup.buttonClicked.connect(self.update_all_data_displays)

        self.actionLoadImage.triggered.connect(self.load_image)

    def start_continuous_shooting(self):
        """Start continuous shooting by spinning up the continuous shooting thread"""
        self.continuousCollectionThread = ContinuousCollectionThread(self.camera_object, self.camera_lock)
        self.continuousCollectionThread.collected.connect(self.process_new_image)
        self.continuousCollectionThread.start()

    def stop_continuous_shooting(self):
        """Stop continuous shooting and wait for the collection thread to finish"""
        self.continuousCollectionThread.stop = True
        # wait until the shooting has stopped, while processing events to keep the app responsive
        while not self.continuousCollectionThread.wait(10):
            QtWidgets.QApplication.processEvents()

    def process_new_image(self, imgdata, image_number=None, burstnumber=None):
        """
        Process a new incoming image so that it is displayed and the data views (fit parameters, intensity profile) are
        updated. Can be connected directly to the collections threads 'collected' signal.

        Args:
            imgdata:        tuple(numpy_image_matrix, jpeg_buffer) containing the new image
            image_number:   the number of this image in the corresponding burst sequence
            burstnumber:    length of the burst sequence this image belongs to
        """
        if image_number is not None and burstnumber is not None:
            self.statusbar.showMessage("Received image {} of {}".format(image_number, burstnumber))
        self.current_image = imgdata[0]
        self.update_all_data_displays()

    def update_all_data_displays(self):
        """
        Update all the data displays: overview and spot previews, intensity profiles and fit parameters.
        """

        # we need to update the images first to let the preview widgets calculate the right crop rectangles
        self.update_images()

        # these methods then use those crop rectangles
        self.plot_intensity_profile()
        self.start_data_fitting()

    def plot_intensity_profile(self):
        """
        Plot the intensity profile across each spot. The profile is plotted along the horizontal and along the vertical
        line that runs through the brightest pixel in the spot crop rectangle.
        """
        try:
            spot_previews = [self.spotAPreviewPlot, self.spotBPreviewPlot]
            intensity_views = [self.spotAIntensity, self.spotBIntensity]
            chart_views = [self.spotAIntensityChartView, self.spotBIntensityChartView]

            # do the same thing for spot A and for spot B
            for i in range(2):
                x1, y1, x2, y2 = spot_previews[i].get_image_rect()

                # cut out the spot rectangle and sum along the color axis to get a 2D semi-grayscale image that
                # can be used to find the brightest pixel quickly. (Note that for real grayscale the colors are
                # weighted - we don't care about that for now)
                spot = np.sum(self.current_image[y1:y2, x1:x2, :], axis=2)

                # find the brightest pixel in the spot crop rectangle
                max_idx = np.argmax(spot)
                max_y, max_x = np.unravel_index(max_idx, spot.shape) + np.array([y1, x1])

                # extract the horizontal and vertical line running through the brightest pixel from the image matrix and
                # apply the selected color processing option
                cpopt = (self.get_cp_option() if
                    self.get_cp_option() != ImageConversion.ColorProcessingOptions.cpColor else
                    ImageConversion.ColorProcessingOptions.cpMonochrome)

                x_line = ImageConversion.apply_color_processing(self.current_image[y1:y2, np.newaxis, max_x, :],
                                                                cpopt, flat=True)
                y_line = ImageConversion.apply_color_processing(self.current_image[np.newaxis, max_y, x1:x2, :],
                                                                cpopt, flat=True)

                # convert sRGB color information to grayscale linear intensity
                x_line_linear = ImageConversion.uint8_color_img_to_linear_intensity_img(x_line).flatten()
                y_line_linear = ImageConversion.uint8_color_img_to_linear_intensity_img(y_line).flatten()

                # create QLineSeries containing the line data, plot the lines on a new QChart and display this QChart in
                # the corresponding QChartView
                line_series_x = QtChart.QLineSeries()
                line_series_y = QtChart.QLineSeries()

                for p in range(len(x_line_linear)):
                    line_series_x.append(float(p), x_line_linear[p])

                for p in range(len(y_line_linear)):
                    line_series_y.append(float(p), y_line_linear[p])

                chart = QtChart.QChart()
                chart.legend().hide()
                chart.addSeries(line_series_x)
                chart.addSeries(line_series_y)
                chart.createDefaultAxes()
                chart.axisY().setRange(0.0, 1.0)

                old_chart = chart_views[i].chart()
                chart_views[i].setChart(chart)
                if old_chart:
                    old_chart.deleteLater()
                intensity_views[i].update()

        except Exception as e:
            self.statusbar.showMessage("[Error while plotting] {}: {}".format(type(e).__name__, str(e)))
            return

    def start_data_fitting(self):
        """Start the data fitting process by spinning up the data fitting thread"""
        if self.get_fitting_method() == SpotFitting.FittingMethods.Disabled:
            # fitting is disabled, get out of here!
            return

        if self.fitDataThread and self.fitDataThread.killable:
            self.fitDataThread.kill = True
            # wait for the thread to finish
            self.fitDataThread.wait()
            print("Fitting process ended")

        self.clear_fit_params()

        spotA_rect = self.spotAPreviewPlot.get_image_rect()
        spotB_rect = self.spotBPreviewPlot.get_image_rect()

        self.fitDataThread = FitDataThread(self.current_image, spotA_rect, spotB_rect, self.get_cp_option(),
                                           self.get_fitting_method(), not self.burst_shooting_on)
        self.fitDataThread.fitted.connect(self.display_fitting_data)

        if self.burst_shooting_on:
            # in the case of a burst, we're also interested in saving the data
            self.fitDataThread.fitted.connect(self.save_fit_data)

        self.fitDataThread.start()

    def display_fitting_data(self, fitparams, fitting_method):
        """
        Display new fitting parameters. Can be directly connected to the fitting thread 'fitted' signal

        Args:
            fitparams: tuple(fitparamsA, fitparamsB) containing the fitted parameters for both spots
        """
        fitparamsA, fitparamsB = fitparams
        labels = SpotFitting.get_fitparam_labels(fitting_method)

        fit_tables = [self.spotAFitTable, self.spotBFitTable]

        print(fitparamsA)
        print(fitparamsB)

        for i in range(2):
            fit_tables[i].setColumnCount(2)
            fit_tables[i].setRowCount(len(labels))
            fit_tables[i].horizontalHeader().hide()
            fit_tables[i].verticalHeader().hide()

            for j in range(len(labels)):
                labelItem = QtWidgets.QTableWidgetItem(labels[j])
                valueItem = QtWidgets.QTableWidgetItem("{:.1f}".format(fitparams[i][j]))

                fit_tables[i].setItem(j, 0, labelItem)
                fit_tables[i].setItem(j, 1, valueItem)

            fit_tables[i].resizeRowsToContents()
            fit_tables[i].setColumnWidth(0, int(fit_tables[i].width() / 2) - 1)
            fit_tables[i].setColumnWidth(1, int(fit_tables[i].width() / 2) - 1)

    def save_new_image(self, imgdata, image_number, burstnumber):
        """
        Save an incoming image according to the selected image storing settings. Can be directly connected to the
        collection thread 'collected' signal

        Args:
            imgdata:        tuple(numpy_image_matrix, jpeg_buffer) containing the new image
            image_number:   the number of this image in the corresponding burst sequence
            burstnumber:    length of the burst sequence this image belongs to
        """
        try:
            if self.save_results:
                if not self.save_crops_only:
                    # save the original image JPEG directly (without conversion to and from an image matrix)
                    filename = "{}orig{:03d}.jpg".format(self.output_prefix, image_number)
                    filepath = os.path.join(self.output_dir, filename)
                    print("Writing image file: {}".format(filepath))
                    jpeg_buffer = imgdata[1]
                    jpeg_buffer.seek(0)
                    with open(filepath, "wb") as f:
                        f.write(jpeg_buffer.getbuffer())

                # crop out and save spot A
                filename = "{}spotA{:03d}.png".format(self.output_prefix, image_number)
                filepath = os.path.join(self.output_dir, filename)
                print("Writing image file: {}".format(filepath))
                cropped_img = ImageConversion.crop_image(imgdata[0], self.spotAPreviewPlot.get_image_rect(),
                                                         return_crop_indices=False)

                pil_img = Image.fromarray(cropped_img)
                with open(filepath, "wb") as f:
                    pil_img.save(f, "png")

                # crop out and save spot B
                filename = "{}spotB{:03d}.png".format(self.output_prefix, image_number)
                filepath = os.path.join(self.output_dir, filename)
                print("Writing image file: {}".format(filepath))
                cropped_img = ImageConversion.crop_image(imgdata[0], self.spotBPreviewPlot.get_image_rect(),
                                                         return_crop_indices=False)

                pil_img = Image.fromarray(cropped_img)
                with open(filepath, "wb") as f:
                    pil_img.save(f, "png")

        except Exception as e:
            self.statusbar.showMessage("[Error] {}: {}".format(type(e).__name__, str(e)))
            return

    def save_fit_data(self, fitparams):
        """
        Save the fit parameters of a single image by appending it to a text file. Can be directly connected to the
        collection thread 'collected' signal.

        Args:
            fitparams: tuple(fitparamsA, fitparamsB) containing the fitted parameters for both spots
        """
        if self.save_results:
            fA, fB = fitparams
            n_fitparams = len(fA)
            dataline = np.zeros((1,n_fitparams * 2))
            dataline[0, :n_fitparams] = fA
            dataline[0, n_fitparams:] = fB
            with open(self.results_file, "ab") as f:
                np.savetxt(f, dataline)

    def burst_finished(self, images_received, all_images_received):
        """
        Clean up after a burst has been finished. Can be directly connected to the burst collection thread 'finished'
        signal

        Args:
            images_received:        total number of images received
            all_images_received:    boolean to indicate whether the number of images received matches the number of
                                    images expected.
        """
        self.statusbar.showMessage("Burst finished after receiving {}/{} images".format(images_received, self.burstnumber))
        self.burst_shooting_on = False
        self.set_camera_controls_enabled(True)
        self.set_burst_controls_enabled(True)
        self.set_spot_controls_enabled(True)
        self.contShootButton.setEnabled(True)
        self.startBurstButton.setEnabled(True)

    def clear_fit_params(self):
        """Clear the tables displaying the fit parameters"""
        self.spotAFitTable.clear()
        self.spotBFitTable.clear()

    def update_images(self):
        """Update the preview images to display the current image"""
        self.cameraPreviewPlot.update_image(self.current_image, cpopt=self.get_cp_option())
        self.spotAPreviewPlot.update_image(self.current_image, cpopt=self.get_cp_option(), center=self.spotAPosition, zoom=self.get_spotA_zoom())
        self.spotBPreviewPlot.update_image(self.current_image, cpopt=self.get_cp_option(), center=self.spotBPosition, zoom=self.get_spotB_zoom())

    def get_spotA_zoom(self):
        """Get the zoom level selected for spot A"""
        slider_val = int(self.spotAZoomSlider.value())
        zoom_val = self.zoom_values[slider_val]
        return zoom_val

    def get_spotB_zoom(self):
        """Get the zoom level selected for spot B"""
        slider_val = int(self.spotBZoomSlider.value())
        zoom_val = self.zoom_values[slider_val]
        return zoom_val

    def get_cp_option(self):
        """Get the selected color processing option"""
        selectedButton = self.colorProcessingGroup.checkedButton()
        if selectedButton == self.cpColor:
            return ImageConversion.ColorProcessingOptions.cpColor
        elif selectedButton == self.cpMonochrome:
            return ImageConversion.ColorProcessingOptions.cpMonochrome
        elif selectedButton == self.cpRed:
            return ImageConversion.ColorProcessingOptions.cpRed
        elif selectedButton == self.cpGreen:
            return ImageConversion.ColorProcessingOptions.cpGreen
        elif selectedButton == self.cpBlue:
            return ImageConversion.ColorProcessingOptions.cpBlue
        return None

    def get_fitting_method(self):
        return SpotFitting.FittingMethods[self.fittingCombo.currentText()]

    def set_camera_controls_enabled(self, state):
        """Enable or disable the camera setting controls"""
        self.shutterSpeedCombo.setEnabled(state)
        self.ISOCombo.setEnabled(state)

    def set_burst_controls_enabled(self, state):
        """Enable or disable the burst setting controls"""
        self.burstnumberInput.setEnabled(state)
        self.saveResultsCB.setEnabled(state)
        self.saveCropsOnlyCB.setEnabled(state)
        self.browseOutputDirButton.setEnabled(state)
        self.outputPrefixInput.setEnabled(state)
        self.fittingCombo.setEnabled(state)

    def set_spot_controls_enabled(self, state):
        """Enable or disable the controls to reposition and change the zoom level of the spots"""
        self.cameraPreviewPlot.setEnabled(state)
        self.spotAPreviewPlot.setEnabled(state)
        self.spotBPreviewPlot.setEnabled(state)
        self.spotAZoomSlider.setEnabled(state)
        self.spotBZoomSlider.setEnabled(state)

        for rb in self.colorProcessingGroup.buttons():
            rb.setEnabled(state)

    def update_zooms(self):
        """
        Read out the zoom level from the zoom sliders, apply them to the spots and redraw everything
        """
        zoom_A = self.get_spotA_zoom()
        zoom_B = self.get_spotB_zoom()

        self.spotAZoomLabel.setText("{:.0f}x".format(zoom_A))
        self.spotBZoomLabel.setText("{:.0f}x".format(zoom_B))

        self.update_all_data_displays()

    def preview_click(self, event):
        """
        Handle click events on the preview widgets and reposition the spots accordingly.
         * left-click on the overview widget repositions spot A
         * right-click on the overview widget repositions spot B
         * any click on the spot A widget repositions spot A
         * any click on the spot B widget repositions spot B
        Args:
            event: mouse click event data
        """
        dpx = np.array([event.x(), event.y()])

        if self.sender() == self.cameraPreviewPlot:
            if event.button() == QtCore.Qt.LeftButton:
                self.spotAPosition = self.cameraPreviewPlot.dpx_to_ipx(dpx)
            elif event.button() == QtCore.Qt.RightButton:
                self.spotBPosition = self.cameraPreviewPlot.dpx_to_ipx(dpx)
        elif self.sender() == self.spotAPreviewPlot:
            self.spotAPosition = self.sender().dpx_to_ipx(dpx)
        elif self.sender() == self.spotBPreviewPlot:
            self.spotBPosition = self.sender().dpx_to_ipx(dpx)

        self.update_all_data_displays()


    def connect_clicked(self):
        """
        Connect the camera and fill the camera setting controls (shutter speed, ISO) with the values that the connected
        camera supports.
        """
        print("Connecting")
        try:
            self.camera_lock.acquire()
            self.camera_object = JsCamera()
            self.camera_object.connect()

            camera_type = self.camera_object.get_camera_type()

            self.connectionStatusLabel.setText("{}".format(camera_type))
            self.statusbar.showMessage("Connected: {}".format(camera_type))

            self.shutterSpeedCombo.clear()
            shutterspeeds = self.camera_object.list_shutterspeed_options()
            cur_shutterspeed = self.camera_object.get_shutterspeed()
            for s in shutterspeeds:
                self.shutterSpeedCombo.addItem(s)
            self.shutterSpeedCombo.setCurrentText(cur_shutterspeed)

            self.ISOCombo.clear()
            isos = self.camera_object.list_iso_options()
            cur_iso = self.camera_object.get_iso()
            for s in isos:
                self.ISOCombo.addItem(s)
            self.ISOCombo.setCurrentText(cur_iso)

            self.connected = True
            self.camera_lock.release()

        except Exception as e:
            self.camera_object = None
            self.statusbar.showMessage("[Error] {}: {}".format(type(e).__name__, str(e)))
            self.camera_lock.release()
            return

        self.connectButton.setEnabled(False)

    def cont_shooting_clicked(self):
        """
        Toggle the continuous shooting state from off to on or vice versa. Block or unblock the camera controls
        accordingly.
        """
        try:
            if not self.connected:
                self.statusbar.showMessage("Not connected")
                return
            # toggle continuous shooting
            self.cont_shooting_on = not self.cont_shooting_on

            if self.cont_shooting_on:
                self.contShootButton.setText("Stop")
                self.startBurstButton.setEnabled(False)
                self.set_camera_controls_enabled(False)
                self.start_continuous_shooting()
                self.statusbar.showMessage("Continuous shooting started")
            else:
                self.contShootButton.setText("Start")
                self.contShootButton.setEnabled(False)
                self.stop_continuous_shooting()
                self.statusbar.showMessage("Continuous shooting stopped")
                self.contShootButton.setEnabled(True)
                self.startBurstButton.setEnabled(True)
                self.set_camera_controls_enabled(True)
        except Exception as e:
            self.statusbar.showMessage("[Error] {}: {}".format(type(e).__name__, str(e)))
            return

    def shutterspeed_changed(self):
        """
        Update the shutter speed value on the camera.
        """
        try:
            if self.connected and self.camera_lock.acquire(block=False):
                self.camera_object.set_shutterspeed(self.shutterSpeedCombo.currentText())
                self.camera_lock.release()
                self.statusbar.showMessage("Shutter speed changed to: {}".format(self.camera_object.get_shutterspeed()))
            else:
                self.statusbar.showMessage("Shutter speed change failed - camera unconnected or in use")
        except Exception as e:
            self.statusbar.showMessage("[Error] {}: {}".format(type(e).__name__, str(e)))
            return

    def iso_changed(self):
        """
        Update the ISO value on the camera
        """
        try:
            if self.connected and self.camera_lock.acquire(block=False):
                self.camera_object.set_iso(self.ISOCombo.currentText())
                self.camera_lock.release()
                self.statusbar.showMessage("ISO changed to: {}".format(self.camera_object.get_iso()))
            else:
                self.statusbar.showMessage("ISO change failed - camera unconnected or in use")
        except Exception as e:
            self.statusbar.showMessage("[Error] {}: {}".format(type(e).__name__, str(e)))
            return

    def browse_output_dir_clicked(self):
        """
        Select an output directory for the results and images to be saved
        """
        current_dir = self.outputDirDisplay.text()
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(parent=self,
                                                                caption="Select output directory",
                                                                directory=current_dir)
        if not output_dir == '':
            self.outputDirDisplay.setText(output_dir)

    def start_burst_clicked(self):
        """
        Start a burst and block the relevant UI controls during the burst.
        """
        try:
            if not self.connected:
                self.statusbar.showMessage("Not connected")
                return

            self.burstnumber = self.burstnumberInput.value()
            self.save_results = self.saveResultsCB.isChecked()
            self.save_crops_only = self.saveCropsOnlyCB.isChecked()
            self.cpopt = self.get_cp_option()

            if self.save_results:
                self.output_prefix = self.outputPrefixInput.text()
                self.output_dir = self.outputDirDisplay.text()

                if not os.path.isdir(self.output_dir):
                    self.statusbar.showMessage("Invalid output directory: {}".format(self.output_dir))
                    return

                # write settings file
                # (this also serves as a check that the selected output directory is writable and the prefix is valid)
                self.settings_file = os.path.join(self.output_dir, (self.output_prefix + "settings.txt"))
                self.write_settings_file(self.settings_file)

                self.results_file = os.path.join(self.output_dir, (self.output_prefix + "results.txt"))
                self.write_results_header(self.results_file)

            self.set_camera_controls_enabled(False)
            self.set_burst_controls_enabled(False)
            self.set_spot_controls_enabled(False)
            self.contShootButton.setEnabled(False)
            self.startBurstButton.setEnabled(False)

            self.burst_shooting_on = True

            self.burstCollectionThread = BurstCollectionThread(self.camera_object, self.camera_lock, self.burstnumber)
            self.burstCollectionThread.collected.connect(self.process_new_image)
            self.burstCollectionThread.collected.connect(self.save_new_image)
            self.burstCollectionThread.finished.connect(self.burst_finished)
            self.burstCollectionThread.start()
        except Exception as e:
            self.statusbar.showMessage("[Error] {}: {}".format(type(e).__name__, str(e)))
            return

    def write_settings_file(self, filename):
        """
        Save current camera settings into a text file.

        Args:
            filename: path to the file to save the settings in
        """
        shutterspeed = self.camera_object.get_shutterspeed()
        iso = self.camera_object.get_iso()
        camera_type = self.camera_object.get_camera_type()
        timestring = datetime.datetime.now().isoformat()
        color_processing = str(self.cpopt)
        spotA_rect = self.spotAPreviewPlot.get_image_rect()
        spotB_rect = self.spotBPreviewPlot.get_image_rect()

        print("Writing settings file: {}".format(filename))
        with open(filename, "w") as f:
            f.write("DIMM measurement started on: \t{}\n".format(timestring))
            f.write("Camera type: \t{}\n".format(camera_type))
            f.write("Shutter speed: \t{}\n".format(shutterspeed))
            f.write("ISO value: \t{}\n".format(iso))
            f.write("Color proc.: \t{}\n".format(color_processing))
            f.write("Spot A rectangle: \t{}\n".format(str(spotA_rect)))
            f.write("Spot B rectangle: \t{}\n".format(str(spotB_rect)))

    def write_results_header(self, filename):
        """
        Create a file to save fitting results in and write the table header.

        Args:
            filename: path to the file to save the results in
        """
        labels = SpotFitting.get_fitparam_labels(self.get_fitting_method())
        headers = [("SpotA." + l) for l in labels] + [("SpotB." + l) for l in labels]
        with open(filename, "wb") as f:
            f.write(b"# DIMM 2D-Gaussian fitting results\n")
            f.write(b"#")
            for h in headers:
                f.write(h.rjust(24).encode("utf-8"))
                f.write(b" ")
            f.write(b"\n")

    def load_image(self):
        """
        Load an image file to be displayed and fitted. Open a file dialog to select the image to be opened
        """
        try:
            image_file, _ = QtWidgets.QFileDialog.getOpenFileName(parent=self, caption="Load image")
            if image_file == '':
                return
            with open(image_file, "rb") as f:
                buffer = BytesIO(f.read())
            image = Image.open(buffer).convert("RGB")
            imgarray = np.array(image)
            imgdata = (imgarray, buffer)
            self.process_new_image(imgdata)

        except Exception as e:
            self.statusbar.showMessage("[Error] {}: {}".format(type(e).__name__, str(e)))
            return

    def save_current_image(self):
        """
        Save the currently loaded image. The image is saved in the current working directory with a name containing the
        current date and time.
        """
        try:
            if self.current_image is not None:
                img = Image.fromarray(self.current_image)
                filename = "saved_images/savedimg_{}.jpg".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                img.save(filename, "jpeg", quality=100)
        except Exception as e:
            self.statusbar.showMessage("[Error] {}: {}".format(type(e).__name__, str(e)))
            return

if __name__ == '__main__':
    # run the application!
    app = QtWidgets.QApplication(sys.argv)
    main = Main()

    main.show()
    sys.exit(app.exec())