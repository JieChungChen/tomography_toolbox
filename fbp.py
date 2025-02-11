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


def proj2split_recon(proj_imgs, angle=1, invert=False):
    if invert:
        proj_imgs = np.invert(proj_imgs)
    sinograms = proj_imgs.transpose(1, 0, 2)

    sp1_recon_stack = []
    sp2_recon_stack = []

    tqdm_range = tqdm(range(len(sinograms)), dynamic_ncols=True)
    for i in tqdm_range:
        tqdm_range.set_description(f"Process projection_{str(i).zfill(3)}")
        im_tmp = sinograms[i]
        im_tmp_roll = np.roll(im_tmp,0)
        sp1 = np.arange(0,im_tmp_roll.shape[0],angle)[1::2] # odd index
        sp2 = np.arange(0,im_tmp_roll.shape[0],angle)[0::2] # even index
        sino_sp1 = np.delete(im_tmp_roll, sp1, 0)
        sino_sp2 = np.delete(im_tmp_roll, sp2, 0)

        vol_geom  = astra.create_vol_geom(proj_imgs.shape[-1])
        proj_geom1 = astra.create_proj_geom('parallel', 1.0, proj_imgs.shape[-1], sp2*np.pi/180)
        proj_geom2 = astra.create_proj_geom('parallel', 1.0, proj_imgs.shape[-1], sp1*np.pi/180)

        sp1_recon = recon_fbp_CUDA(vol_geom,proj_geom1,sino_sp1)
        sp2_recon = recon_fbp_CUDA(vol_geom,proj_geom2,sino_sp2)

        sp1_recon_stack.append(sp1_recon)
        sp2_recon_stack.append(sp2_recon)

    # return as 8 bit images
    sp1_recon_stack = np.array(sp1_recon_stack)
    sp2_recon_stack = np.array(sp2_recon_stack)
    sp1_recon_stack_tmp = min_max_normalize(sp1_recon_stack)*255
    sp1_recon_stack_tmp = np.round(sp1_recon_stack_tmp).astype(np.uint8)
    sp2_recon_stack_tmp = min_max_normalize(sp2_recon_stack)*255
    sp2_recon_stack_tmp = np.round(sp2_recon_stack_tmp).astype(np.uint8)
    return sp1_recon_stack_tmp, sp2_recon_stack_tmp


def sino_to_slice(sino, angles):
    vol_geom  = astra.create_vol_geom(sino.shape[-1])
    proj_geom = astra.create_proj_geom('parallel', 1.0, sino.shape[-1], angles)
    recon = recon_fbp_CUDA(vol_geom, proj_geom, sino)
    recon = min_max_normalize(recon)*255
    recon = np.round(recon).astype(np.uint8)
    return recon


if __name__ == '__main__':
    folder_path = 'JIFaproma_python/projections_test' 
    files = sorted(glob.glob(folder_path+"/*"))
    save_dir = 'reconstruct_test'

    proj_imgs = []
    for f in files:
        im = Image.open(f)
        ### nn ### 
        # im = im-im.min()
        # im = im/im.max()
        # im = np.uint8((im*255).round())
        ### n ###
        # im = np.float32(im/im.max())    
        # im = np.uint8((im.clip(0, 1)*255.).round())
        proj_imgs.append(im)
    proj_imgs = np.array(proj_imgs)

    print(proj_imgs.dtype, proj_imgs.max(), proj_imgs.min(), proj_imgs.shape, type(proj_imgs))

    save_path1 = f'JIFaproma_python/{save_dir}_sp1'
    os.makedirs(f'JIFaproma_python/{save_dir}_sp1', exist_ok=True)
    save_path2 = f'JIFaproma_python/{save_dir}_sp2'
    os.makedirs(f'JIFaproma_python/{save_dir}_sp2', exist_ok=True)

    sp1, sp2 = proj2split_recon(proj_imgs)

    for j in range(len(sp1)):
        cv2.imwrite(f'{save_path1}/{save_dir}_{str(j+1).zfill(4)}-han.png', sp1[j])
        cv2.imwrite(f'{save_path2}/{save_dir}_{str(j+1).zfill(4)}-han.png', sp2[j])