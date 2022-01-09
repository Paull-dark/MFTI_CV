# File for visualizing Digital Imaging in Medicine (DICOM)
# voi - volume of interst
# lut - look up table
# https://www.medicalconnections.co.uk/kb/Lookup-Tables
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def dicom2array(path, voi_lut=True, fix_monochrome=True):

    '''
    Function is called for converting dicom format to numpy array
    :param path: path of the pics
    :param voi_lut: The Value Of Interest(VOI) LUT transformation
    transforms the modality pixel values into pixel values which are
    meaningful for the user or the application.
    In other words, VOI LUT (if available by DICOM device) is used to
    transform raw DICOM data to "human-friendly" view
    :param fix_monochrome:
    :return: np.ndarray
    '''

    dicom = pydicom.read_file(path)
    if voi_lut:
        data=apply_voi_lut(dicom.pixel_array, dicom)
    else:
        dara = dicom.pixel_array

    # depending on this value, X-ray scan might look inverted. Fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def plot_imgs(imgs, cols=4, size=7, title="", cmap='gray', img_size=(500,500)):
    '''
    function is called to plot pictures
    '''
    rows = len(imgs)//cols + 1
    fig = plt.figure(figsize=(cols*size, rows*size))
    for i, img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout(pad=0.5)
    plt.show()