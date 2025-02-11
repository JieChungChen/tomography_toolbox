import os
import astra
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm


def min_max_normalize(arr):
    return (arr-arr.min())/(arr.max()-arr.min())


def recon_fbp_CUDA(vol_geom,proj_geom,sino_sp):
    """
    Reconstruct a 2D slice from a sinogram using the filtered back-projection (FBP) method with CUDA acceleration.
    vol_geom: geometry of a reconstructed slice
    proj_geom: geometry of a parallel projection
    sino_sp: sinogram
    """
    sino_id = astra.data2d.create('-sino', proj_geom, sino_sp)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    cfg['option'] = { 'FilterType': 'hann' }
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    recon = astra.data2d.get(rec_id)

    sino_0 = np.ones((sino_sp.shape[0],sino_sp.shape[-1]))
    s0_id = astra.data2d.create('-sino', proj_geom, sino_0)
    rec0_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ProjectionDataId'] = s0_id
    cfg['ReconstructionDataId'] = rec0_id
    cfg['option'] = { 'FilterType': 'hann' }
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    recon0 = astra.data2d.get(rec0_id)

    recon_n = recon/recon0

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(rec0_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(s0_id)
    return recon_n


def sino_to_slice(sino, angles):
    vol_geom  = astra.create_vol_geom(sino.shape[-1])
    proj_geom = astra.create_proj_geom('parallel', 1.0, sino.shape[-1], angles)
    recon = recon_fbp_CUDA(vol_geom, proj_geom, sino)
    recon = min_max_normalize(recon)*255
    recon = np.round(recon).astype(np.uint8)
    return recon
