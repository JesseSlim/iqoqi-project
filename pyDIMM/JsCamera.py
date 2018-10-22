"""
DSLR camera control module

Author: Jesse Slim, December 2016
"""


import numpy as np
import gphoto2 as gp
from PIL import Image
from io import BytesIO
import os, time


class JsCamera:
    """
    Represents a connected DSLR, and provides communication using libgphoto2 internally.

    Tailored to the Nikon D3s, but adapting it for other cameras shouldn't be too hard
    Possible concerns when adapting to another camera:
     * configuration keys may have different names: check the result of cli 'gphoto2 --auto-detect --list-config'
     * burst capture may be done in a different way: e.g. for Canon DSLRs using config eosremoterelease
     * shutter speeds may be represented differently: '0.004s' for Nikon DSLRs, '1/2500' for Canon DSLRs
    """
    CFG_ShutterSpeed = "shutterspeed"
    CFG_BurstNumber = "burstnumber"
    CFG_Manufacturer = "manufacturer"
    CFG_Model = "cameramodel"
    CFG_ISO = "iso"

    # maximum waiting interval when waiting for the capturing to finish
    timeout_capture = 15000

    # maximum waiting interval when waiting for the next file to become available
    timeout_file = 5000

    def __init__(self):
        """
        Initialize the JsCamera object and the libgphoto2 library
        """
        self.context = gp.Context()
        self.camera = gp.Camera()

    def __del__(self):
        """
        Delete the JsCamera object, disconnecting the camera
        """
        self.disconnect()

    def connect(self):
        """
        Open the connection to the camera

        Raises:
            GPhoto2Error ([-105] Unknown model): if no camera can be found
        """

        # to function on OS X, the PTP camera deamon, claiming the PTP camera USB port, has to be stopped
        os.popen("killall PTPCamera")
        self.camera.init(self.context)

    def disconnect(self):
        """Close the connection to the camera"""
        self.camera.exit(self.context)

    def print_summary(self):
        """
        Print the summary of the camera settings provided by the camera

        Returns:
            None
        """
        text = self.camera.get_summary(self.context)
        print(str(text))

    def get_camera_type(self):
        """Return a string representing the manufacturer and the model of the camera"""
        manufacturer = self.get_config_value(self.CFG_Manufacturer)
        model = self.get_config_value(self.CFG_Model)

        return "{} {}".format(manufacturer, model)

    def set_config_value(self, key, value):
        """
        Set a configuration key to the value supplied. See the libgphoto2/gphoto2 documentation for information on
        configuration keys.

        Args:
            key:    string representing the config key to be changed
            value:  new value for the config key

        Returns:
            None
        """
        config_tree = self.camera.get_config(self.context)
        config_key = config_tree.get_child_by_name(key)

        config_key.set_value(value)
        self.camera.set_config(config_tree, self.context)

    def get_config_value(self, key):
        """
        Get the value of a configuration key. See the libgphoto2/gphoto2 documentation for information on configuration
        keys

        Args:
            key: string representing the config key to be read

        Returns:
            the current value of the configuration key
        """
        config_tree = self.camera.get_config(self.context)
        config_key = config_tree.get_child_by_name(key)

        return config_key.get_value()

    def list_radio_config_options(self, key):
        """
        List the possible values for a configuration key of the 'radio' type. See the libgphoto2/gphoto2 documentation
        for information on configuration keys.

        Args:
            key: string representing the config key to be examined

        Returns:
            list containing the possible radio values
        """
        config_tree = self.camera.get_config(self.context)
        config_key = config_tree.get_child_by_name(key)

        if not config_key.get_type() == gp.GP_WIDGET_RADIO:
            raise ValueError("Config setting '" + key + "' is not a radio setting")

        num_options = config_key.count_choices()
        options = list()

        for i in range(num_options):
            options.append(config_key.get_choice(i))

        return options

    def find_closest_shutterspeed_option(self, shutterspeed):
        """
        Find the shutter speed setting closest to the desired setting.
        Args:
            shutterspeed: float representing the desired shutter time in seconds

        Returns:
            (value_string, actual_shutterspeed) representing the configuration value string closest to the desired
            setting and a floating point indicating the closest shutterspeed in seconds
        """

        # note that this function is Nikon-specific
        # other cameras may have other ways of representing shutter times in the 'shutterspeed' config radio options.
        # this can easily be found out using the gphoto2 command-line interface:
        # the command 'gphoto2 --auto-detect --get-config shutterspeed' should list all available shutter speeds and their
        # representation, e.g. 0.004s for Nikon DSLRs vs 1/250 for Canon DSLRs
        options = self.list_shutterspeed_options()
        candidate_deviation = np.inf
        candidate = None
        candidate_speed = None
        for i, value in options.items():
            try:
                fval = float(value.replace("s", ""))
                deviation = np.abs(shutterspeed - fval)
                if (deviation < candidate_deviation):
                    candidate = i
                    candidate_speed = fval
                    candidate_deviation = deviation
            except ValueError:
                pass

        return options[candidate], candidate_speed

    def list_shutterspeed_options(self):
        """
        List all shutter speed options supported by the connected camera.

        Returns:
            list containing the configuration value strings for all shutter speeds
        """
        options = self.list_radio_config_options(self.CFG_ShutterSpeed)
        return options

    def list_iso_options(self):
        """
        List all ISO options supported by the connected camera.

        Returns:
            list containing the configuration value strings for all ISO options.
        """
        options = self.list_radio_config_options(self.CFG_ISO)
        return options

    def set_shutterspeed(self, option):
        """Set the shutter speed to the given shutter speed configuration value string."""
        self.set_config_value(self.CFG_ShutterSpeed, option)

    def get_shutterspeed(self):
        """Get the configuration value string for the current shutter speed."""
        return self.get_config_value(self.CFG_ShutterSpeed)

    def set_burstnumber(self, number):
        """Set the burst number."""
        self.set_config_value(self.CFG_BurstNumber, number)

    def get_burstnumber(self):
        """Get the burst number."""
        return self.get_config_value(self.CFG_BurstNumber)

    def set_iso(self, option):
        """Set the ISO option to the given ISO option configuration value string."""
        self.set_config_value(self.CFG_ISO, option)

    def get_iso(self):
        """Get the configuration value string for the current ISO option."""
        return self.get_config_value(self.CFG_ISO)

    def trigger_capture(self):
        """Trigger a capture on the connected camera. Note that this does not retrieve the image."""
        self.camera.trigger_capture(self.context)

    def collect_all_images(self, callback, colormode="RGB"):
        """
        Collect all available images on the camera one by one until a timeout is reached.

        Args:
            callback:   function to be called upon the retrieval of each image. The function will be called with the
                        numpy image matrix as argument
            colormode:  color conversion to be applied by Pillow to the incoming image. See the Pillow documentation for
                        possible values. Most notably: "RGB" to get color images, "L" to get grayscale images.

        Returns:
            None
        """
        # start an event loop that collects the incoming files
        i = 0
        first_wait = True
        while True:
            timeout = self.timeout_file
            # allow the first timeout to be longer - the camera may only start uploading files after capturing has been
            # completed
            if first_wait == True:
                timeout = self.timeout_capture
                first_wait = False

            res = self.check_and_collect_image(timeout, colormode)

            if res:
                i += 1
                # call the callback with the imgarray as argument
                callback(res[0])
            elif res == None:
                # another event occurred, ignore
                pass
            elif res == False:
                print("Timeout occurred - ending collection loop")
                break

    def check_and_collect_image(self, timeout, colormode):
        """
        Check if a single image becomes available within the specified timeout, and if so, collect and return it.

        Args:
            timeout:    timeout in milliseconds to wait for an image to become available
            colormode:  color conversion to be applied by Pillow to the incoming image. See the Pillow documentation for
                        possible values. Most notably: "RGB" to get color images, "L" to get grayscale images.

        Returns:
            * False if a timeout occurs
            * None if another camera event occurs before an image becomes available
            * tuple(numpy_image_matrix, jpeg_buffer) if an image is retrieved
        """
        type, data = self.camera.wait_for_event(timeout, self.context)
        if type == gp.GP_EVENT_FILE_ADDED:
            print("File added: " + data.folder + data.name)
            camerafile = self.camera.file_get(data.folder, data.name, gp.GP_FILE_TYPE_NORMAL, self.context)

            print(".. downloading data")
            data = camerafile.get_data_and_size()

            print(".. decoding JPEG")
            buffer = BytesIO(data)
            img = Image.open(buffer).convert(colormode)
            imgarray = np.array(img)

            print(".. done, returning data")
            # callback(imgarray)

            # return the data
            return (imgarray, buffer)

        elif type == gp.GP_EVENT_TIMEOUT:
            # print("Timeout occurred")
            return False
        else:
            # print("Other event occurred: " + str(type))
            return None

    def capture_sequence_and_collect_images(self, number, callback=None, shutterspeed=None, colormode="RGB"):
        """
        Start the capture of a sequence of photos and collect them one by one.

        Args:
            number:         number of photos to be taken
            callback:       function to be called upon the retrieval of each image. The function will be called with the
                            numpy image matrix as argument
            shutterspeed:   desired shutter speed, if None the current shutter speed is used
            colormode:      color conversion to be applied by Pillow to the incoming image. See the Pillow documentation
                            for possible values. Most notably: "RGB" to get color images, "L" to get grayscale images.

        Returns:
            None
        """
        if shutterspeed:
            shutter_setting, _ = self.find_closest_shutterspeed_option(shutterspeed)
            self.set_shutterspeed(shutter_setting)

        self.set_burstnumber(number)
        self.trigger_capture()

        if callback:
            self.collect_all_images(callback, colormode=colormode)