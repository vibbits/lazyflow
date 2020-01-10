###############################################################################
# VIB, Frank Vernaillen, Jan 2020
# Implementation based on lazyflow\lazyflow\classifiers\tiktorchLazyflowClassifier
###############################################################################
from builtins import range
import pickle as pickle
import tempfile

import numpy
import logging
import sys

# import GPUtil


from .lazyflowClassifier import LazyflowPixelwiseClassifierABC

from neuralnets.util.tools import load_net
from neuralnets.util.validation import segment

from skimage.external import tifffile

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DeepLearningLazyflowClassifier(LazyflowPixelwiseClassifierABC):

    def __init__(self, net, filename=None, batch_size=1, window_size=256):
        logger.debug(f"DeepLearningLazyflowClassifier __init__ net={net} filename={filename} window_size={window_size} batch_size={batch_size}")
        # GPUtil.showUtilization()

        self._filename = filename
        if self._filename is None:
            self._filename = ""

        self.batch_size = batch_size
        self.window_size = (window_size, window_size)

        if net is None:  # CHECKME: does this ever happen??
            print(self._filename)
            # tiktorch_net = TikTorch.unserialize(self._filename)
            net = load_net(self._filename)

        self._net = net

        self._image_nr = 0  # for debugging only, counts the number of tiles that we were fed for prediction and saved to disk for analysis

    def predict_probabilities_pixelwise(self, image, roi, axistags=None):  # CHECKME: where does this get called - how do we see that image is actually just the original image?

        num_channels = len(self.known_classes)
        # expected_shape = [stop - start for start, stop in zip(roi[0], roi[1])] + [num_channels]

        # TODO: check that ROI starts at 0,0,0 and is as large as the input image
        # TODO: check that axistags is zyxc or tyxc; for other tag organizations we could use OpReorderAxes (see tiktorch lazyfloz classifier)
        # TODO: check that num_channels == 1 (or that it is identical to the # input channels of the net)

        #tifffile.imsave(f"c:\\users\\frankvn\\development\\ilastik_vib_{self._image_nr}.tif", image.astype("uint16"))
        self._image_nr += 1


        logger.debug(f"DeepLearningLazyFlowClassifier.predict_probabilities_pixelwise(): nr={self._image_nr} image.shape={image.shape} roi={list(roi)} known_classes={self.known_classes} axistags={''.join(axistags.keys())}")

        # In our examples so far, image is (num z, height, width, num channels = 1) in our examples; however this could be different if axistags != zyxc

        input_data = image[:, :, :, 0]  # shape should be (num z slices, image height, image width)
        # Note: we expect input_data.shape[0] == self.BATCH_SIZE, except in case # slices in the full image stack is not a multiple of self.BATCH_SIZE (because of the blockshape setting in opDLclass.py)

        logger.debug(f"neuralnets.segment: input_data shape={input_data.shape} min={input_data.min():.2f}, max={input_data.max():.2f}, mean={input_data.mean():.2f}; window_size={self.window_size} batch_size={self.batch_size}")
        # GPUtil.showUtilization()
        try:
            # Ask neural net for class probability.
            segmented_data = segment(input_data, self._net, self.window_size, self.batch_size, step_size=None, train=False)
        except Exception as ex:
            # An exception occurred. A CUDA out of memory error, for example.
            logger.critical(ex)
            return None  # FIXME: return a result array with all zeros instead. Or is None handled correctly on the receiving end?

        # The neural net returned only the probability a pixel is "foreground" (e.g. part of a mitochondrion).
        # but Ilastik expects a probability for each class. So generate that desired output.
        assert num_channels == 2, 'VibDeepLearningLazyFlowClassifier is a binary classifier'
        result = numpy.stack((1-segmented_data, segmented_data), axis=-1)  # this has shape (z, y, x, 2), the last dimension being background/foreground class
        logger.debug(f"neuralnets.segment: segmented_data shape={segmented_data.shape} min={segmented_data.min():.2f}, max={segmented_data.max():.2f}, mean={segmented_data.mean():.2f}; result shape={result.shape}")
        # GPUtil.showUtilization()

        return result

    @property
    def known_classes(self):
        return list(range(self._net.out_channels))

    @property
    def feature_count(self):
        return self._net.in_channels

    def get_halo_shape(self, data_axes="zyxc"):
        # Halo's are not needed/supported for this classifier.
        if len(data_axes) == 4:
            return (0, 0, 0, 0)
        elif len(data_axes) == 3:
            return (0, 0, 0)

    def serialize_hdf5(self, h5py_group):
        logger.debug("DeepLearningLazyFlowClassifier.serialize_hdf5() - skipped!!")
        # logger.debug("Serializing")
        # h5py_group[self.HDF5_GROUP_FILENAME] = self._filename
        # h5py_group["pickled_type"] = pickle.dumps(type(self), 0)
        #
        # # HACK: can this be done more elegantly?
        # with tempfile.TemporaryFile() as f:
        #     self._tiktorch_net.serialize(f)
        #     f.seek(0)
        #     h5py_group["classifier"] = numpy.void(f.read())

    @classmethod
    def deserialize_hdf5(cls, h5py_group):
        logger.debug("DeepLearningLazyFlowClassifier.deserialize_hdf5() - skipped!!")
        # # TODO: load from HDF5 instead of hard coded path!
        # logger.debug("Deserializing")
        # # HACK:
        # # filename = PYTORCH_MODEL_FILE_PATH
        # filename = h5py_group[cls.HDF5_GROUP_FILENAME]
        # logger.debug("Deserializing from {}".format(filename))
        #
        # with tempfile.TemporaryFile() as f:
        #     f.write(h5py_group["classifier"].value)
        #     f.seek(0)
        #     loaded_pytorch_net = TikTorch.unserialize(f)
        #
        # return DeepLearningLazyflowClassifier(loaded_pytorch_net, filename)
        return None


assert issubclass(DeepLearningLazyflowClassifier, LazyflowPixelwiseClassifierABC)
