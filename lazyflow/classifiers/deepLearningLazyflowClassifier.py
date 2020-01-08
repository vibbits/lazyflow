###############################################################################
# VIB, Frank Vernaillen, Jan 2020
# Implementation based on lazyflow\lazyflow\classifiers\tiktorchLazyflowClassifier
###############################################################################
from builtins import range
import pickle as pickle
import tempfile


import numpy
import vigra
import random

from .lazyflowClassifier import LazyflowPixelwiseClassifierABC, LazyflowPixelwiseClassifierFactoryABC
from lazyflow.operators.opReorderAxes import OpReorderAxes
from lazyflow.graph import Graph
from lazyflow.roi import roiToSlice

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import sys
sys.path.append(r'E:\git\bits\bioimaging\deep_segment\neuralnets')  # FIXME - can we avoid this?
print(f'sys.path={sys.path}')
from neuralnets.util.tools import load_net
from neuralnets.util.validation import segment


# FIXME: hard coded file path to a trained and pickled pytorch network!
# PYTORCH_MODEL_FILE_PATH = '/Users/chaubold/opt/miniconda/envs/ilastik-py3/src/tiktorch/test3.nn'
# PYTORCH_MODEL_FILE_PATH = '/Users/jmassa/Downloads/dnunet-cpu-chaubold.nn'

VIB_MODEL_FILE_PATH = r"E:\git\bits\bioimaging\deep_segment\pretrained_models\dense\epfl_vnc_mira\unet_multi_domain\best_checkpoint.pytorch"  # Path to pre-trained model

# class DeepLearningLazyflowClassifierFactory(LazyflowPixelwiseClassifierFactoryABC):
#     # The version is used to determine compatibility of pickled classifier factories.
#     # You must bump this if any instance members are added/removed/renamed.
#     VERSION = 1
#
#     def __init__(self, *args, **kwargs):
#         self._args = args
#         self._kwargs = kwargs
#
#         logger.debug("DeepLearningLazyflowClassifierFactory __init__()")
#
#         print(self._args)
#
#         # FIXME: hard coded file path to a trained and pickled pytorch network!
#         self._filename = None  # self._args[0]
#         self._loaded_net = None
#
#     def create_and_train_pixelwise(self, feature_images, label_images, axistags=None, feature_names=None):
#         self._filename = VIB_MODEL_FILE_PATH  # was: PYTORCH_MODEL_FILE_PATH
#         logger.debug(f"DeepLearningLazyflowClassifierFactory create_and_train_pixelwise() - actually just loading network from {self._filename}")
#
#         # Save for future reference
#         # known_labels = numpy.sort(vigra.analysis.unique(y))
#
#         # TODO: check whether loaded network has the same number of classes as specified in ilastik!
#         self._loaded_net = load_net(self._filename)
#         logger.info(self.description)
#
#         # logger.info("OOB during training: {}".format( oob ))
#         return DeepLearningLazyflowClassifierFactory(self._loaded_net, self._filename)
#
#     def get_halo_shape(self, data_axes="zyxc"):
#         logger.debug(f"DeepLearningLazyflowClassifierFactory get_halo_shape()")
#         # return (z_halo, y_halo, x_halo, 0)
#         if len(data_axes) == 4:
#             return (0, 32, 32, 0)
#         # FIXME: assuming 'yxc' !
#         elif len(data_axes) == 3:
#             return (32, 32, 0)
#
#     @property
#     def description(self):
#         logger.debug(f"DeepLearningLazyflowClassifierFactory description()")
#         if self._loaded_net:
#             return (
#                 f"network loaded from {self._filename} with "
#                 f"input channels = {self._loaded_net.in_channels} and "
#                 f"output channels = {self._loaded_net.out_channels}"
#             )
#         else:
#             return f"network loading from {self._filename} failed"
#
#     def estimated_ram_usage_per_requested_predictionchannel(self):
#         # FIXME: compute from model size somehow??
#         return numpy.inf
#
#     def __eq__(self, other):
#         return isinstance(other, type(self)) and self._args == other._args and self._kwargs == other._kwargs
#
#     def __ne__(self, other):
#         return not self.__eq__(other)
#
#
# assert issubclass(DeepLearningLazyflowClassifierFactory, LazyflowPixelwiseClassifierFactoryABC)


class DeepLearningLazyflowClassifier(LazyflowPixelwiseClassifierABC):
    # HDF5_GROUP_FILENAME = "pytorch_network_path"

    def __init__(self, net, filename=None, HALO_SIZE=32, BATCH_SIZE=3):
        """
        Args:
            net (xxx): xxx object to be loaded into this
              classifier object
            filename (None, optional): Save file name for future reference
        """
        logger.debug(f"DeepLearningLazyflowClassifier __init__ net={net} filename={filename} HALO_SIZE={HALO_SIZE} BATCH_SIZE={BATCH_SIZE}")
        self._filename = filename
        if self._filename is None:
            self._filename = ""

        self.HALO_SIZE = HALO_SIZE
        self.BATCH_SIZE = BATCH_SIZE

        if net is None:
            print(self._filename)
            # tiktorch_net = TikTorch.unserialize(self._filename)
            net = load_net(self._filename)

        # print (self._filename)

        # assert tiktorch_net.return_hypercolumns == False
        # print('blah')

        self._net = net

        # self._opReorderAxes = OpReorderAxes(graph=Graph())
        # self._opReorderAxes.AxisOrder.setValue("zcyx")

    def predict_probabilities_pixelwise(self, feature_image, roi, axistags=None):
        """
        Implicitly assumes that feature_image is includes the surrounding HALO!
        roi must be chosen accordingly
        """
        taskid = random.randint(1, 1000000)  # hack. Due to concurrency output of multiple tasks gets mixed. By adding the taskid to our output we can match the correct snippets of output.

        num_channels = len(self.known_classes)

        logger.debug(f"taskid={taskid} DeepLearningLazyFlowClassifier.predict_probabilities_pixelwise(): feature_image.shape={feature_image.shape} roi={list(roi)} known_classes={self.known_classes} axistags={''.join(axistags.keys())}")

        expected_shape = [stop - start for start, stop in zip(roi[0], roi[1])] + [num_channels]  # CHECK, what does this do?

        # note: the given 'roi' parameter indicates the required shape of the result
        # FIXME: need to take ROI into account?
        assert num_channels == 2, 'VibDeepLearningLazyFlowClassifier is a binary classifier'
        result_shape = feature_image.shape[0:3]   #  feature_image is (num z, height, width, num feature channels)  I think; for now we're building a hack, so ignore num feature channels
        result_shape = (*result_shape, num_channels)
        result = numpy.zeros(result_shape, dtype=float)

        input_data = feature_image[:, :, :, 0]  # shape should be (num z slices, image height, image width)
        patch_size = (input_data.shape[1], input_data.shape[2])  # CHECKME ask joris: how to pick patch size   # FIXME - what are restrictions on patch size? does it need to be square? rectangular? power of 2?
        if (patch_size[0] % 64 == 0) and (patch_size[1] % 64 == 0):
            logger.debug(f"taskid={taskid} expected_shape={expected_shape}; calling joris net segment: input_data shape={input_data.shape} min={input_data.min():.2f}, max={input_data.max():.2f}, mean={input_data.mean():.2f}")

            try:
                # Ask neural net for class probability.
                segmented_data = segment(input_data, self._net, patch_size, batch_size=1, step_size=None, train=False)
            except Exception as ex:
                # An exception occurred. A CUDA out of memory error, for example.
                logger.critical(ex)
                return result

            # The neural net returned only the probability a pixel is "foreground" (e.g. part of a mitochondrion).
            # but Ilastik expects a probability for each class. So generate that desired output.
            result[:, :, :, 1] = segmented_data
            result[:, :, :, 0] = 1.0 - result[:, :, :, 1]
            logger.debug(f"taskid={taskid} Stats of joris net: shape={segmented_data.shape} min={segmented_data.min():.2f}, max={segmented_data.max():.2f}, mean={segmented_data.mean():.2f}; Stats of result: shape={result.shape} min={result.min():.2f}, max={result.max():.2f}, mean={result.mean():.2f}")
        else:
            logger.debug(f'taskid={taskid} SKIPPED PATCH WITH ANNOYING SIZE; expected_shape={expected_shape} input_data shape={input_data.shape}')

        return result

    # def predict_probabilities_pixelwise(self, feature_image, roi, axistags=None):
    #     """
    #     Implicitly assumes that feature_image is includes the surrounding HALO!
    #     roi must be chosen accordingly
    #     """
    #     logger.info(f"predicting using pytorch network for image of shape {feature_image.shape} and roi {roi}")
    #     logger.info(
    #         f"Stats of input: min={feature_image.min()}, max={feature_image.max()}, mean={feature_image.mean()}"
    #     )
    #     logger.info(f"expected pytorch input shape is {self._tiktorch_net.expected_input_shape}")
    #     logger.info(f"expected pytorch output shape is {self._tiktorch_net.expected_output_shape}")
    #
    #     # print(self._tiktorch_net.expected_input_shape)
    #     # print(self._tiktorch_net.expected_output_shape)
    #
    #     num_channels = len(self.known_classes)
    #     expected_shape = [stop - start for start, stop in zip(roi[0], roi[1])] + [num_channels]
    #
    #     self._opReorderAxes.Input.setValue(vigra.VigraArray(feature_image, axistags=axistags))
    #     self._opReorderAxes.AxisOrder.setValue("zcyx")
    #     reordered_feature_image = self._opReorderAxes.Output([]).wait()
    #
    #     # normalizing patch
    #     # reordered_feature_image = (reordered_feature_image - reordered_feature_image.mean()) / (reordered_feature_image.std() + 0.000001)
    #
    #     if len(self._tiktorch_net.get("window_size")) == 2:
    #         exp_input_shape = numpy.array(self._tiktorch_net.expected_input_shape)
    #         exp_input_shape = tuple(numpy.append(1, exp_input_shape))
    #         print(exp_input_shape)
    #     else:
    #         exp_input_shape = self._tiktorch_net.expected_input_shape
    #
    #     logger.info(
    #         f"input axistags are {axistags}, "
    #         f"Shape after reordering input is {reordered_feature_image.shape}, "
    #         f"axistags are {self._opReorderAxes.Output.meta.axistags}"
    #     )
    #
    #     slice_shape = list(reordered_feature_image.shape[1::])  # ignore z axis
    #     # assuming [z, y, x]
    #     result_roi = numpy.array(roi)
    #     if slice_shape != list(exp_input_shape[1::]):
    #         logger.info(f"Expected input shape is {exp_input_shape[1::]}, " f"but got {slice_shape}, reshaping...")
    #
    #         # adding a zero border to images that have the specific shape
    #
    #         exp_shape = list(self._tiktorch_net.expected_input_shape[1::])
    #         zero_img = numpy.zeros(exp_shape)
    #
    #         # diff shape: cyx
    #         diff_shape = numpy.array(exp_input_shape[1::]) - numpy.array(slice_shape)
    #         # diff_shape = numpy.array(self._tiktorch_net.expected_input_shape) - numpy.array(slice_shape)
    #         # offset shape z, y, x, c for easy indexing, with c = 0, z = 0
    #         offset = numpy.array([0, 0, 0, 0])
    #         logger.info(f"Diff_shape {diff_shape}")
    #
    #         # at least one of y, x (diff_shape[1], diff_shape[2]) should be off
    #         # let's determine how to adjust the offset -> offset[2] and offset[3]
    #         # caveat: this code assumes that image requests were tiled in a regular
    #         # pattern starting from left upper corner.
    #         # We use a blocked array-cache to achieve that
    #         # y-offset:
    #         if diff_shape[1] > 0:
    #             # was the halo added to the upper side of the feature image?
    #             # HACK: this only works because we assume the data to be in zyx!!!
    #             if roi[0][1] == 0:
    #                 # no, doesn't seem like it
    #                 offset[1] = self.HALO_SIZE
    #
    #         # x-offsets:
    #         if diff_shape[2] > 0:
    #             # was the halo added to the upper side of the feature image?
    #             # HACK: this only works because we assume the data to be in zyx!!!
    #             if roi[0][2] == 0:
    #                 # no, doesn't seem like it
    #                 offset[2] = self.HALO_SIZE
    #
    #         # HACK: still assuming zyxc
    #         result_roi[0] += offset[0:3]
    #         result_roi[1] += offset[0:3]
    #         reorder_feature_image_extents = numpy.array(reordered_feature_image.shape)
    #         # add the offset:
    #         reorder_feature_image_extents[2:4] += offset[1:3]
    #         # zero_img[:, :, offset[1]:reorder_feature_image_extents[2], offset[2]:reorder_feature_image_extents[3]] = \
    #         #     reordered_feature_image
    #
    #         # reordered_feature_image = zero_img
    #
    #         pad_img = numpy.pad(
    #             reordered_feature_image,
    #             [
    #                 (0, 0),
    #                 (0, 0),
    #                 (offset[1], exp_input_shape[2] - reorder_feature_image_extents[2]),
    #                 (offset[2], exp_input_shape[3] - reorder_feature_image_extents[3]),
    #             ],
    #             "reflect",
    #         )
    #
    #         reordered_feature_image = pad_img
    #
    #         logger.info(f"New Image shape {reordered_feature_image.shape}")
    #
    #     result = numpy.zeros([reordered_feature_image.shape[0], num_channels] + list(reordered_feature_image.shape[2:]))
    #
    #     logger.info(f"forward")
    #
    #     # we always predict in 2D, per z-slice, so we loop over z
    #     for z in range(0, reordered_feature_image.shape[0], self.BATCH_SIZE):
    #         # logger.warning("Dumping to {}".format('"/Users/chaubold/Desktop/dump.h5"'))
    #         # vigra.impex.writeHDF5(reordered_feature_image[z,...], "data", "/Users/chaubold/Desktop/dump.h5")
    #
    #         # create batch of desired num slices. Multiple slices can be processed on multiple GPUs!
    #         batch = [
    #             reordered_feature_image[zi : zi + 1, ...].reshape(self._tiktorch_net.expected_input_shape)
    #             for zi in range(z, min(z + self.BATCH_SIZE, reordered_feature_image.shape[0]))
    #         ]
    #         logger.info(f"batch info: {[x.shape for x in batch]}")
    #
    #         print("batch info:", [x.shape for x in batch])
    #
    #         # if len(self._tiktorch_net.get('window_size')) == 2:
    #         #     print("BATTCHHHHH", batch.shape)
    #
    #         result_batch = self._tiktorch_net.forward(batch)
    #         logger.info(f"Resulting slices from {z} to {z + len(batch)} have shape {result_batch[0].shape}")
    #
    #         print("Resulting slices from ", z, " to ", z + len(batch), " have shape ", result_batch[0].shape)
    #
    #         for i, zi in enumerate(range(z, (z + len(batch)))):
    #             result[zi : (zi + 1), ...] = result_batch[i]
    #
    #     logger.info(f"Obtained a predicted block of shape {result.shape}")
    #
    #     print("Obtained a predicted block of shape ", result.shape)
    #
    #     self._opReorderAxes.Input.setValue(vigra.VigraArray(result, axistags=vigra.makeAxistags("zcyx")))
    #     # axistags is vigra.AxisTags, but opReorderAxes expects a string
    #     self._opReorderAxes.AxisOrder.setValue("".join(axistags.keys()))
    #     result = self._opReorderAxes.Output([]).wait()
    #     logger.info(f"Reordered result to shape {result.shape}")
    #
    #     # FIXME: not needed for real neural net results:
    #     logger.info(f"Stats of result: min={result.min()}, max={result.max()}, mean={result.mean()}")
    #
    #     # cut out the required roi
    #     logger.info(f"Roi shape {result_roi}")
    #
    #     # crop away halo and reorder axes to match "axistags"
    #     # crop in X and Y:
    #     cropped_result = result[roiToSlice(*result_roi)]
    #
    #     logger.info(f"cropped the predicted block to shape {cropped_result.shape}")
    #
    #     return cropped_result

    @property
    def known_classes(self):
        # return list(range(self._tiktorch_net.expected_output_shape[0]))
        return list(range(self._net.out_channels))

    @property
    def feature_count(self):
        # return self._tiktorch_net.expected_input_shape[0]
        return self._net.in_channels

    def get_halo_shape(self, data_axes="zyxc"):
        if len(data_axes) == 4:
            return (0, self.HALO_SIZE, self.HALO_SIZE, 0)
        # FIXME: assuming 'yxc' !
        elif len(data_axes) == 3:
            return (self.HALO_SIZE, self.HALO_SIZE, 0)

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
