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



from .lazyflowClassifier import LazyflowPixelwiseClassifierABC

from neuralnets.util.tools import load_net
from neuralnets.util.validation import segment

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DeepLearningLazyflowClassifier(LazyflowPixelwiseClassifierABC):

    def __init__(self, net, filename, batch_size=1, window_size=256):
        # If 'net' is not None, then it will be used. Otherwise, the neural network is loaded from 'filename'.
        logger.debug(f"DeepLearningLazyflowClassifier __init__ net={net} filename={filename} window_size={window_size} batch_size={batch_size}")
        # GPUtil.showUtilization()

        self.batch_size = batch_size
        self.window_size = (window_size, window_size)

        if filename is None:
            self._filename = ""
        else:
            self._filename = filename

        if net is None:
            self._net = load_net(self._filename)
        else:
            self._net = net

    def predict_probabilities_pixelwise(self, image, roi, axistags=None):
        # CHECKME: where does predict_probabilities_pixelwise() get called?   From OpPixelwiseClassifierPredict._calculate_probabilities(roi), called from OpBaseClassifierPredict.execute(slot, subindex, roi, result)
        # Where do we see that 'image' here is actually just the original image and not feature images calculated from it?

        # Check that the neural network expects grayscale images.
        assert self._net.in_channels == 1

        # We asked Ilastik (by setting a specific 'blockshape' on 'opDLclass' in dlClassGui) to feed us images at the
        # full width and height (but possibly fewer z-slices than in the full image stack).
        # We do not want a halo either. Check that the roi agrees with this assumption.
        # So the ROI should be [[z_start 0 0], [z_end image_height image_width]].
        assert (roi[0][1] == 0) and (roi[0][2] == 0)
        assert (roi[1][1] == image.shape[1]) and (roi[1][2] == image.shape[2])

        # The number of z-planes we are given is normally equal to the batch size that the user requested, except for
        # the last batch since there may be fewer z-planes left.
        assert image.shape[0] <= self.batch_size

        # Check that axistags is zyxc or tyxc. For other data organizations we could actually use OpReorderAxes
        # (see tiktorch lazyflow classifier) but we leave that for future work...
        axistags_str = ''.join(axistags.keys())
        if axistags_str != 'zyxc' and axistags_str != 'tyxc':
            logger.critical(f"The image axes are '{axistags_str}' but we only support 'zyxc' or 'tyxc'. "
                            "Please reorder the axes in Ilastik after loading the image.")
            return numpy.zeros((*image.shape[0:3], 2))

        # logger.debug(f"DeepLearningLazyFlowClassifier.predict_probabilities_pixelwise(): image.shape={image.shape} roi={list(roi)} axistags={''.join(axistags.keys())}")

        # Here 'image' is (num z, height, width, num channels = 1). We just reshape it to 'input_data' of shape
        # (num z, height, width) since that is what the neural network expects.
        input_data = image[:, :, :, 0]

        # logger.debug(f"neuralnets.segment: input_data shape={input_data.shape} min={input_data.min():.2f}, max={input_data.max():.2f}, mean={input_data.mean():.2f}; window_size={self.window_size} batch_size={self.batch_size}")

        try:
            # Ask neural net for class probability (for each z-slice in input_data)
            segmented_data = segment(input_data, self._net, self.window_size, self.batch_size, step_size=None, train=False)
        except Exception as ex:
            # An exception occurred. This could be a CUDA out of memory error, for example.
            logger.critical(ex)
            return numpy.zeros((*image.shape[0:3], 2))

        # The neural net returned only the probability a pixel is "foreground" (e.g. part of a mitochondrion).
        # but Ilastik expects a probability for each class. So generate that desired output.
        num_classes = len(self.known_classes)
        assert num_classes == 2, 'VibDeepLearningLazyFlowClassifier is a binary classifier: background/foreground'

        # The desired output is the probability for each of the two pixel classes (foreground/background).
        # Build it from the foreground probability we got from the neural network.
        # 'result' will have shape (z, y, x, 2), the last dimension being background/foreground class
        result = numpy.stack((1-segmented_data, segmented_data), axis=-1)

        # logger.debug(f"neuralnets.segment: segmented_data shape={segmented_data.shape} min={segmented_data.min():.2f}, max={segmented_data.max():.2f}, mean={segmented_data.mean():.2f}; result shape={result.shape}")

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
        logger.warning("DeepLearningLazyFlowClassifier.serialize_hdf5() - skipped!!")
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
        logger.warning("DeepLearningLazyFlowClassifier.deserialize_hdf5() - skipped!!")
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
