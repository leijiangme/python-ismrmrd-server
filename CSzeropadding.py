import ismrmrd
import os
import itertools
import logging
import traceback
import numpy as np
import numpy.fft as fft
import xml.dom.minidom
import base64
import ctypes
import re
import mrdhelper
import constants
from time import perf_counter


# Folder for debug output files
def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("A matrix size of (%s x %s x %s) and a field of view of (%s x %s x %s)mm^3",
                     metadata.encoding[0].encodedSpace.matrixSize.x,
                     metadata.encoding[0].encodedSpace.matrixSize.y,
                     metadata.encoding[0].encodedSpace.matrixSize.z,
                     metadata.encoding[0].encodedSpace.fieldOfView_mm.x,
                     metadata.encoding[0].encodedSpace.fieldOfView_mm.y,
                     metadata.encoding[0].encodedSpace.fieldOfView_mm.z)
    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages

    acqGroup = []
    acqGroup_Cali = []

    try:
        for item in connection:
            if isinstance(item, ismrmrd.Acquisition):
                if (not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) and
                        not item.is_flag_set(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA) and
                        item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)):
                    acqGroup_Cali.append(item)
                elif (not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) and
                      not item.is_flag_set(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA) and
                      not item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)):
                    acqGroup.append(item)

                if item.head.flagIsSet(ismrmrd.ACQ_LAST_IN_SLICE):
                    if item.head.flagIsSet(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                        image = process_raw(acqGroup_Cali, connection, config, metadata)
                        acqGroup_Cali = []  # reset
                    # connection.send_image(image)
                    else:
                        image = process_raw(acqGroup, connection, config, metadata)
                        acqGroup = []  # reset

                    connection.send_image(image)

            elif item is None:  # Receive MRD_MESSAGE_CLOSE
                break
            else:
                # logging.error("Unhandled data type: %s", class(item))
                continue

        # Process remaining items if any
        if len(acqGroup_Cali) > 0:
            logging.info("Untriggered: Processing calibration data now")
            image = process_raw(acqGroup_Cali, connection, config, metadata)
            connection.send_image(image)
            acqGroup_Cali = []

        if len(acqGroup) > 0:
            logging.info("Untriggered: Processing raw data now")
            image = process_raw(acqGroup, connection, config, metadata)
            connection.send_image(image)
            acqGroup = []
    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())
    finally:
        connection.send_close()


def process_raw(group, connection, config, metadata):
    logging.info("Config: %s\n", config)

    full_x_matrix = metadata.encoding.encodedSpace.matrixSize.x  # including oversampling x 2
    full_x_sample = group
    {1}.head.number_of_samples;
    x_center_sample = group
    {1}.head.center_sample;

    # Use the first acquisition to determine the number of readout points and coils
    if floor(full_x_sample / 2.0) == x_center_sample:
        readout_num = size(group
        {1}.data, 1)
        else:
        readout_num = full_x_matrix

    coil_num = size(group
    {1}.data, 2);
    # Total number of phase encoding lines (obtained from metadata or prior knowledge)
    PE_num = metadata.encoding.encodingLimits.kspace_encoding_step_1.maximum + 1;
    # Prepare k_space_matrix with all potential phase encoding lines
    k_space_matrix = zeros(readout_num, coil_num, PE_num, 'like', group
    {1}.data);
    # logging.info("readout_num: #d", readout_num);
    # logging.info("PE_num: %d", PE_num);

    # Determine the indexing range based on x_center_sample condition
    if floor(full_x_sample / 2.0) == x_center_sample:
        row_range = 1:readout_num;
    else:
        row_range = (full_x_matrix / 2.0 - x_center_sample + 1):full_x_matrix;

    # Reconstruct images
    images = []

    # Loop through each acquisition and add it to the correct k-space matrix
    for i = 1:numel(group)
    acq = group
    {i};
    k_space_matrix(row_range,:, acq.head.idx.kspace_encode_step_1 + 1) = acq.data;


# Format data into a single [RO PE cha] array
# ksp = cell2mat(permute(cellfun(@(x) x.data, group, 'UniformOutput', false), [1 3 2]));

ksp = permute(k_space_matrix, [1 3 2]);

# Fourier Transform
for n=1:size(ksp, 3)
img(:,:, n) = fftshift(ifft2(ifftshift(ksp(:,:, n))));


# Sum of squares coil combination
img = sqrt(sum(abs(img). ^ 2, 3));

# Remove phase oversampling
im = img(round(size(img, 1) / 4 + 1):round(size(img, 1) * 3 / 4),:);

im = im. * (32767. / max(im(:)));
im = int16(round(im));

# Create MRD Image object, set image data and (matrix_size, channels, and data_type) in header
image = ismrmrd.Image(im);

# Copy the relevant AcquisitionHeader fields to ImageHeader
# image.head = image.head.fromAcqHead(group{centerIdx}.head);
image.head = image.head.fromAcqHead(group
{1}.head);

# field_of_view is mandatory
image.head.field_of_view = single([metadata.encoding(1).reconSpace.fieldOfView_mm.x...
                                  metadata.encoding(1).reconSpace.fieldOfView_mm.y...
                                  metadata.encoding(1).reconSpace.fieldOfView_mm.z]);

# Set ISMRMRD Meta Attributes
meta = struct;
meta.DataRole = 'Image';
meta.ProcessingHistory = 'MATLAB';
meta.WindowCenter = uint16(16384);
meta.WindowWidth = uint16(32768);
meta.ImageRowDir = group
{1}.head.read_dir;
meta.ImageColumnDir = group
{1}.head.phase_dir;

# set_attribute_string also updates attribute_string_len
image = image.set_attribute_string(ismrmrd.Meta.serialize(meta));

# Append
images
{end + 1} = image;
