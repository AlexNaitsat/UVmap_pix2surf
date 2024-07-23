import torch
import numpy as np


def load_multiview_batch(in_batch):
    device = torch.device("cuda")
    nox_v = [item.float().permute(0, 3, 1, 2).to(device) for item in in_batch['nox-v']]  # [0,1]
    nox_x = [item.float().permute(0, 3, 1, 2).to(device) for item in in_batch['nox-x']]  # [0,1]
    rgb_v = [item.float().permute(0, 3, 1, 2).to(device) for item in in_batch['rgb-v']]  # [0,1]
    rgb_x = [item.float().permute(0, 3, 1, 2).to(device) for item in in_batch['rgb-x']]  # [0,1]
    mask_v = [item.float().permute(0, 3, 1, 2).to(device) for item in in_batch['mask-v']]  # 0,1
    mask_x = [item.float().permute(0, 3, 1, 2).to(device) for item in in_batch['mask-x']]  # 0,1

    uv_v = [item.float().permute(0, 2, 1).unsqueeze(3).to(device) for item in in_batch['uv-v']]
    uv_x = [item.float().permute(0, 2, 1).unsqueeze(3).to(device) for item in in_batch['uv-x']]
    uv_mask_v = [item.float().permute(0, 2, 1).unsqueeze(3).to(device) for item in in_batch['uv-mask-v']]
    uv_mask_x = [item.float().permute(0, 2, 1).unsqueeze(3).to(device) for item in in_batch['uv-mask-x']]
    uv_xyz_v = [item.float().permute(0, 2, 1).unsqueeze(3).to(device) for item in in_batch['uv-xyz-v']]
    uv_xyz_x = [item.float().permute(0, 2, 1).unsqueeze(3).to(device) for item in in_batch['uv-xyz-x']]

    crr_idx_mtx = [[ii.long().permute(0, 2, 1).unsqueeze(3).to(device) for ii in item]
                   for item in in_batch['crr-idx-mtx']]
    crr_mask_mtx = [[ii.float().permute(0, 2, 1).unsqueeze(3).to(device) for ii in item]
                    for item in in_batch['crr-mask-mtx']]

    pack = {'rgb-v': rgb_v, 'rgb-x': rgb_x, 'nox-v': nox_v, 'nox-x': nox_x,
            'mask-v': mask_v, 'mask-x': mask_x,
            'uv-v': uv_v, 'uv-x': uv_x,
            'uv-mask-v': uv_mask_v, 'uv-mask-x': uv_mask_x,
            'uv-xyz-v': uv_xyz_v, 'uv-xyz-x': uv_xyz_x,
            'crr-idx-mtx': crr_idx_mtx, 'crr-mask-mtx': crr_mask_mtx}
    return {'to-nn': pack, 'meta-info': in_batch['info']}


def load_singleview_batch(in_batch):#prepare the batch for network forward: load array in GPU and convert remained RGB values into [0,1] range
    device = torch.device("cuda")
    nox_v = in_batch['nox-v'].float().permute(0, 3, 1, 2).to(device) / 255.0  # [0,1]
    nox_x = in_batch['nox-x'].float().permute(0, 3, 1, 2).to(device) / 255.0  # [0,1]
    rgb_v = in_batch['rgb-v'].float().permute(0, 3, 1, 2).to(device) / 255.0  # [0,1]
    rgb_x = in_batch['rgb-x'].float().permute(0, 3, 1, 2).to(device) / 255.0  # [0,1]
    mask_v = in_batch['mask-v'].float().permute(0, 3, 1, 2).to(device)  # 0,1
    mask_x = in_batch['mask-x'].float().permute(0, 3, 1, 2).to(device)  # 0,1

    uv_v = in_batch['uv-v'].float().permute(0, 2, 1).unsqueeze(3).to(device)
    uv_x = in_batch['uv-x'].float().permute(0, 2, 1).unsqueeze(3).to(device)
    uv_mask_v = in_batch['uv-mask-v'].float().permute(0, 2, 1).unsqueeze(3).to(device)
    uv_mask_x = in_batch['uv-mask-x'].float().permute(0, 2, 1).unsqueeze(3).to(device)
    uv_xyz_v = in_batch['uv-xyz-v'].float().permute(0, 2, 1).unsqueeze(3).to(device) / 255
    uv_xyz_x = in_batch['uv-xyz-x'].float().permute(0, 2, 1).unsqueeze(3).to(device) / 255
    pack = {'rgb-v': rgb_v, 'rgb-x': rgb_x, 'nox-v': nox_v, 'nox-x': nox_x,
            'mask-v': mask_v, 'mask-x': mask_x,
            'uv-v': uv_v, 'uv-x': uv_x,
            'uv-mask-v': uv_mask_v, 'uv-mask-x': uv_mask_x,
            'uv-xyz-v': uv_xyz_v, 'uv-xyz-x': uv_xyz_x}
    return {'to-nn': pack, 'meta-info': in_batch['info']}


def spread_feature(container, learned_uv, feature, mask1c):
    """
    :param container: B,C,R,R
    :param learned_uv: B,2,H,W
    :param feature: B,C,H,W aligned with latent uv map
    :param mask1c: B,1,H,W used to mask latent uv and feature
    :return: container
    """
    assert float(mask1c.max()) < (1.0 + 1e-9)
    assert container.shape[1] == feature.shape[1]
    c = container.shape[1]
    res = container.shape[2]
    _learned_uv = learned_uv * mask1c.repeat(1, 2, 1, 1)
    _feature = feature * mask1c.repeat(1, c, 1, 1)
    learned_uv = torch.clamp((_learned_uv * res).long(), 0, res - 1)
    learned_uv = learned_uv.reshape(learned_uv.shape[0], 2, -1)
    learned_uv = learned_uv[:, 0, :] * res + learned_uv[:, 1, :]  # B, R*R
    learned_uv = learned_uv.unsqueeze(1).repeat(1, c, 1)  # B,C,R*R
    container = container.reshape(container.shape[0], container.shape[1], -1)
    container = container.scatter(2, learned_uv, _feature.reshape(feature.shape[0], c, -1))
    container = container.reshape(container.shape[0], container.shape[1], res, res)
    return container

# query_uv is array of
def query_feature(feature_map, query_uv):
    """
    query features from feature map
    :param feature_map: B,C,res1,res2
    :param query_uv: B,2,K,1 in [0,1]
    :return B,C,K,1
    """
    assert float(query_uv.max()) < 1 + 1e-9
    assert query_uv.shape[1] == 2
    res1 = feature_map.shape[2]
    res2 = feature_map.shape[3]
    query_index = query_uv.clone()
    query_index[:, 0, ...] = torch.clamp((query_uv[:, 0, ...] * res1).long(), 0, res1 - 1)
    query_index[:, 1, ...] = torch.clamp((query_uv[:, 1, ...] * res2).long(), 0, res2 - 1)
    if query_index.ndimension() > 3:
        index = query_index.squeeze(3)  # B,2,K
    else:
        index = query_index  # B*2*K
    # cvt to 1D index
    index = index[:, 0, :] * feature_map.shape[3] + index[:, 1, :]  # B,K
    index = index.unsqueeze(2).repeat(1, 1, feature_map.shape[1])
    flatten_feature_map = feature_map.reshape(feature_map.shape[0],
                                              feature_map.shape[1], -1).permute(0, 2, 1)  # B,N,C
    query = torch.gather(flatten_feature_map, 1, index.long()).contiguous()  # B,K,C
    query = query.permute(0, 2, 1).unsqueeze(3)
    return query


def make_grid(res, return_np=False):
    dim0 = np.arange(0, res[0]) + 0.5
    dim0 = dim0 / len(dim0)
    dim1 = np.arange(0, res[1]) + 0.5
    dim1 = dim1 / len(dim1)
    col_uv, row_uv = np.meshgrid(dim1, dim0)
    super_uv = np.concatenate((row_uv[..., np.newaxis], col_uv[..., np.newaxis]), 2)  # R,R,2
    super_uv_tensor = torch.from_numpy(super_uv.astype(np.float32))
    if return_np:
        return super_uv_tensor.permute(2, 0, 1).unsqueeze(0), super_uv
    else:
        return super_uv_tensor.permute(2, 0, 1).unsqueeze(0)

def output_network_dict(D, prefix=''):
    if prefix =='':
        print(f'{prefix}{type(D)}')
    prefix = prefix + '  '
    for name in D.keys():
        D_value = D[name]
        if isinstance(D_value, list):
            print(f'{prefix}[{name}]  :  {type(D_value)}')
            output_tensor_list(D_value, prefix)
        elif isinstance(D_value, dict):
            print(f'{prefix}[{name}]  :  {type(D_value)}')
            output_network_dict(D_value, prefix)
        elif hasattr(D_value, 'shape'):
            print(f'{prefix}[{name}]  : {D_value.shape}')
        else:
            print(f'{prefix}[{name}]  :  {type(D_value)}')

def output_tensor_list(L, prefix=''):
    if prefix == '':
        print(f'{prefix}{type(L)}')
    prefix = prefix + '     '
    for indx in range(0, len(L)):
        if isinstance(L[indx], list):
            print(f'{prefix}[{indx}]  :  {type(L[indx])}')
            output_tensor_list(L[indx], prefix)
        elif isinstance(L[indx], dict):
            print(f'{prefix}[{indx}]  :  {type(L[indx])}')
            output_network_dict(L[indx], prefix)
        elif  hasattr(L[indx], 'shape'):
            print(f'{prefix}[{indx}]  :  {L[indx].shape}')
        else:
            print(f'{prefix}[{indx}]  :  {type(L[indx])}')

#t = torch.Size([batch_n, color_n=3, W=240, H=320])
def tensor_to_batch(T, prefix, to_normalize=False):
    import cv2 as cv
    if torch.is_tensor(T):
        T = T.detach().cpu().numpy()
    if to_normalize:
        maxT =np.max(T)
        if maxT != 0: T= T/maxT
    batch_n = T.shape[0]
    for b in range(0,batch_n):
        image_array = np.transpose(T[b,:,:,:], [1, 2, 0])
        if  np.max(image_array) < 1.0001:
            image_array = image_array.astype('float32')*255 #convert to RGB range
        image_file_name = './debug/' + prefix + '_batch_' + str(b) +'.png'
        print('writing ' + image_file_name)
        cv.imwrite(image_file_name, image_array)

def draw_multiview_tensor_with_batchs(Tlist,  prefix):
    import cv2 as cv
    view_num = len(Tlist)
    batch_n = Tlist[0].shape[0]
    H  = Tlist[0].shape[2]
    W  = Tlist[0].shape[3]
    mv_images_per_batch = [np.zeros((H,W*view_num,3)).astype('float32')]*batch_n
    for b in range(0, batch_n):
        for v in range(0, view_num):
            T = Tlist[v]
            if torch.is_tensor(T):
                T = T.detach().cpu().numpy()
                image_array = np.transpose(T[b,:,:,:], [1, 2, 0])
                if  np.max(image_array) < 1.0001:
                    image_array = image_array.astype('float32')*255 #convert to RGB range
                mv_images_per_batch[b][:, W*v:W*(v+1),:] = image_array
        image_file_name = './debug/' + prefix + '_batch_' + str(b) + '.png'
        print('writing ' + image_file_name)
        cv.imwrite(image_file_name, mv_images_per_batch[b])
