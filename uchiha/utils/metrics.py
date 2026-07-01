import cv2
import numpy as np
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
import scipy
from PIL import Image
import scipy.ndimage
import scipy.special
import math
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

import skimage.measure

from uchiha.utils.data import normalize


def calculate_psnr_ssim(target, pred):
    """
    计算 PSNR 和 SSIM
    Args:
        target (ndarray): HWC 形状, 真实值
        pred (ndarray): HWC 形状, 预测值
    Returns:
        psnr (float)
        ssim (float)
    """

    # 1. 计算 PSNR
    data_range = target.max() - target.min()
    psnr_val = psnr_func(target, pred, data_range=data_range)

    # 2. 计算 SSIM
    # channel_axis=2 表示通道在最后一个维度 (H, W, C)
    ssim_val = ssim_func(target, pred, data_range=data_range, channel_axis=2)

    return psnr_val, ssim_val


def calculate_uqi(target, pred):
    """
    计算 UQI (Universal Quality Image Index)
    UQI 是 SSIM 的一种特殊情况（不含亮度/对比度常数），
    通常对每个波段计算后取平均。
    """

    def _uqi_single_channel(t, p):
        # 展平以便计算统计量
        t = t.flatten()
        p = p.flatten()

        mx = np.mean(t)
        my = np.mean(p)

        # 样本协方差与方差 (使用 N-1 或 N 均可，保持一致即可，这里用 numpy 默认)
        cov_xy = np.cov(t, p)[0][1]
        var_x = np.var(t)
        var_y = np.var(p)

        # 避免分母为 0
        eps = 1e-8

        # UQI 公式
        numerator = 4 * cov_xy * mx * my
        denominator = (var_x + var_y) * (mx ** 2 + my ** 2) + eps

        return numerator / denominator

    # 逐通道计算
    channels = target.shape[2]
    uqis = []
    for i in range(channels):
        uqis.append(_uqi_single_channel(target[:, :, i], pred[:, :, i]))

    return np.mean(uqis)


def calculate_sam(target, pred):
    """
    计算 SAM (Spectral Angle Mapper)
    Args:
        target: (H, W, C)
        pred: (H, W, C)
    Returns:
        mean_sam (float): 平均光谱角（单位：角度 degree）
    """
    # 确保没有 0 向量，避免除以 0
    eps = 1e-8

    # 在通道维度 (axis=2) 上计算点积
    # dot_product shape: (H, W)
    dot_product = np.sum(target * pred, axis=2)

    # 计算范数 (L2 norm)
    # norm shape: (H, W)
    norm_target = np.linalg.norm(target, axis=2)
    norm_pred = np.linalg.norm(pred, axis=2)

    # 计算余弦值
    denominator = norm_target * norm_pred + eps
    cos_theta = dot_product / denominator

    # 截断数值以防数值误差导致 arccos 越界
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 计算角度，论文表格采用 SAM (°)，因此将 arccos 的弧度结果转换为角度。
    sam_map = np.degrees(np.arccos(cos_theta))

    # 如果有的像素全是0，sam_map 可能是 nan，将其置为 0
    sam_map = np.nan_to_num(sam_map)

    return np.mean(sam_map)


def calculate_ag(img):
    """
    计算 AG (Average Gradient) - 无参考指标
    衡量图像的清晰度/纹理丰富程度
    """
    # img shape: (H, W, C)
    # 计算 x 和 y 方向的梯度
    # 使用简单的差分或者是 Sobel 算子。AG 标准定义常用简单的差分。

    # 逐通道计算，最后取平均
    img = normalize(img)
    channels = img.shape[2]
    ag_vals = []

    for c in range(channels):
        band = img[:, :, c]
        sobelx = cv2.Sobel(band, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=3)
        ag_val = np.mean(np.sqrt(sobelx ** 2 + sobely ** 2))
        ag_vals.append(ag_val)

    return np.mean(ag_vals)


def _as_quality_hwc(img):
    img = np.nan_to_num(np.asarray(img, dtype=np.float32), copy=False)
    if img.ndim == 2:
        return img[:, :, None]
    if img.ndim != 3:
        raise ValueError(f'Expected a 2D or 3D image for quality metrics, got shape {img.shape}')
    return img


def _quality_normalize(img):
    img = _as_quality_hwc(img)
    return normalize(img)


gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range)
a *= a
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)


def aggd_features(imdata):
    # flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt / right_mean_sqrt
    else:
        gamma_hat = np.inf
    # solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata)) ** 2) / (np.average(imdata2))
    else:
        r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    # solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm) ** 2);
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    # mean parameter
    N = (br - bl) * (gam2 / gam1)  # *aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)


def ggd_features(imdata):
    nr_gam = 1 / prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq / E ** 2
    pos = np.argmin(np.abs(nr_gam - rho));
    return gamma_range[pos], sigma_sq


def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image ** 2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image ** 2))
    return (image - mu_image) / (var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl + br) / 2.0,
                     alpha1, N1, bl1, br1,  # (V)
                     alpha2, N2, bl2, br2,  # (H)
                     alpha3, N3, bl3, bl3,  # (D1)
                     alpha4, N4, bl4, bl4,  # (D2)
                     ])


def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)


def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)


def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = int(patch_size)
    patches = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)

    patches = np.array(patches)

    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features


def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)
    img2 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)

    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size / 2)

    feats = np.hstack((feats_lvl1, feats_lvl2))  # feats_lvl3))

    return feats


def calculate_niqe(inputImgData):
    inputImgData = normalize(inputImgData)
    patch_size = 96
    module_path = dirname(__file__)

    # TODO: memoize
    params = scipy.io.loadmat(join(module_path, 'niqe_image_params.mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]

    M, N, C = inputImgData.shape

    # assert C == 1, "niqe called with videos containing %d channels. Please supply only the luminance channel" % (C,)
    assert M > (
            patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (
            patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

    if inputImgData.max() <= 1.1:
        inputImgData = inputImgData * 255.0
    niqe_list = []

    for c in range(C):
        channel_data = inputImgData[:,:,c]
        feats = get_patches_test_features(channel_data, patch_size)
        sample_mu = np.mean(feats, axis=0)
        sample_cov = np.cov(feats.T)

        X = sample_mu - pop_mu
        covmat = ((pop_cov + sample_cov) / 2.0)
        pinvmat = scipy.linalg.pinv(covmat)
        niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))
        niqe_list.append(niqe_score)

    return np.mean(niqe_list)


def _brisque_channel_features(channel_data):
    channel_data = np.nan_to_num(channel_data.astype(np.float32), copy=False)
    if channel_data.max() <= 1.1:
        channel_data = channel_data * 255.0

    features = []
    current = channel_data
    for _ in range(2):
        mscn, _, _ = compute_image_mscn_transform(current)
        alpha, _, _, _, left_sqrt, right_sqrt = aggd_features(mscn.copy())
        features.extend([alpha, (left_sqrt ** 2 + right_sqrt ** 2) / 2.0])

        for product in paired_product(mscn):
            alpha, mean_param, _, _, left_sqrt, right_sqrt = aggd_features(product.copy())
            features.extend([alpha, mean_param, left_sqrt ** 2, right_sqrt ** 2])

        if min(current.shape) < 32:
            current = current
        else:
            current = cv2.resize(current, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    return np.asarray(features, dtype=np.float32)


def _brisque_feature_score(features):
    # BRISQUE normally feeds this NSS vector to a trained SVR. The project does
    # not ship an SVR model, so we use a deterministic naturalness-distance
    # surrogate that preserves the BRISQUE convention: lower is better.
    reference = []
    scale = []
    for _ in range(2):
        reference.extend([2.0, 1.0])
        scale.extend([1.5, 1.0])
        for _ in range(4):
            reference.extend([1.0, 0.0, 0.5, 0.5])
            scale.extend([1.5, 1.0, 1.0, 1.0])

    reference = np.asarray(reference, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    distance = np.sqrt(np.mean(((features - reference) / (scale + 1e-8)) ** 2))
    return float(np.clip(100.0 * (1.0 - np.exp(-distance / 2.0)), 0.0, 100.0))


def calculate_brisque(inputImgData):
    """
    Calculate a BRISQUE-style no-reference score for HSI data.

    The score is based on two-scale MSCN/AGGD NSS features. Since this project
    does not include a trained BRISQUE SVR model file, the final prediction is a
    bounded naturalness-distance score in the standard lower-is-better range.
    """
    inputImgData = _quality_normalize(inputImgData)
    brisque_list = []
    for c in range(inputImgData.shape[2]):
        features = _brisque_channel_features(inputImgData[:, :, c])
        brisque_list.append(_brisque_feature_score(features))
    return float(np.mean(brisque_list))


def _piqe_block_has_artifact(block, threshold=0.1):
    segments = []
    for edge in (block[0, :], block[-1, :], block[:, 0], block[:, -1]):
        edge = np.asarray(edge, dtype=np.float32)
        if edge.size <= 6:
            segments.append(edge)
        else:
            for start in range(0, edge.size - 5, 5):
                segments.append(edge[start:start + 6])
    return min(float(np.std(segment)) for segment in segments) < threshold


def _piqe_block_has_noise(block):
    sigma = float(np.std(block))
    h, w = block.shape
    h0, h1 = h // 4, h - h // 4
    w0, w1 = w // 4, w - w // 4
    center = block[h0:h1, w0:w1]
    mask = np.ones(block.shape, dtype=bool)
    mask[h0:h1, w0:w1] = False
    surround = block[mask]
    center_std = float(np.std(center)) + 1e-8
    surround_std = float(np.std(surround)) + 1e-8
    ratio = surround_std / center_std
    beta = abs(sigma - ratio) / (max(sigma, ratio) + 1e-8)
    return sigma > 2.0 * beta


def _piqe_channel_score(channel_data, block_size=16, activity_threshold=0.1,
                        artifact_threshold=0.1):
    channel_data = np.nan_to_num(channel_data.astype(np.float32), copy=False)
    if channel_data.max() <= 1.1:
        channel_data = channel_data * 255.0

    mscn, _, _ = compute_image_mscn_transform(channel_data)
    height, width = mscn.shape
    height = (height // block_size) * block_size
    width = (width // block_size) * block_size
    if height < block_size or width < block_size:
        return 0.0
    mscn = mscn[:height, :width]

    active_blocks = 0
    distortion = 0.0
    for top in range(0, height, block_size):
        for left in range(0, width, block_size):
            block = mscn[top:top + block_size, left:left + block_size]
            block_var = float(np.var(block))
            if block_var <= activity_threshold:
                continue

            active_blocks += 1
            normalized_var = float(np.clip(block_var, 0.0, 1.0))
            if _piqe_block_has_artifact(block, threshold=artifact_threshold):
                distortion += 1.0 - normalized_var
            if _piqe_block_has_noise(block):
                distortion += normalized_var

    if active_blocks == 0:
        return 0.0
    return float(np.clip(100.0 * (distortion + 1.0) / (active_blocks + 1.0), 0.0, 100.0))


def calculate_piqe(inputImgData):
    """
    Calculate PIQE no-reference score by block-wise distortion estimation.

    The score follows the lower-is-better [0, 100] convention and is averaged
    over all HSI bands.
    """
    inputImgData = _quality_normalize(inputImgData)
    piqe_list = []
    for c in range(inputImgData.shape[2]):
        piqe_list.append(_piqe_channel_score(inputImgData[:, :, c]))
    return float(np.mean(piqe_list))


def _safe_ggd_features(data):
    data = np.nan_to_num(np.asarray(data, dtype=np.float32).reshape(-1),
                         nan=0.0, posinf=0.0, neginf=0.0)
    if data.size == 0:
        return 2.0, 0.0
    sigma_sq = float(np.var(data))
    abs_mean = float(np.mean(np.abs(data)))
    if sigma_sq <= 1e-12 or abs_mean <= 1e-12:
        return 2.0, sigma_sq

    rho = sigma_sq / (abs_mean ** 2 + 1e-12)
    nr_gam = 1.0 / prec_gammas
    pos = int(np.argmin(np.abs(nr_gam - rho)))
    return float(gamma_range[pos]), sigma_sq


def _safe_aggd_features(data):
    data = np.nan_to_num(np.asarray(data, dtype=np.float32).copy(),
                         nan=0.0, posinf=0.0, neginf=0.0)
    if data.size == 0 or float(np.var(data)) <= 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    alpha, mean_param, left_beta, right_beta, _, _ = aggd_features(data)
    values = np.nan_to_num(
        np.asarray([alpha, mean_param, left_beta ** 2, right_beta ** 2], dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0)
    return tuple(float(v) for v in values)


def _safe_weibull_features(data):
    data = np.abs(np.nan_to_num(np.asarray(data, dtype=np.float32).reshape(-1),
                                nan=0.0, posinf=0.0, neginf=0.0))
    if data.size == 0:
        return 1.0, 0.0
    mean_val = float(np.mean(data))
    std_val = float(np.std(data))
    if mean_val <= 1e-12 or std_val <= 1e-12:
        return 10.0, mean_val

    coeff_var = std_val / (mean_val + 1e-12)
    shape = float(np.clip(coeff_var ** -1.086, 0.1, 10.0))
    scale = mean_val / (float(scipy.special.gamma(1.0 + 1.0 / shape)) + 1e-12)
    return shape, float(scale)


def _iter_quality_patches(image, patch_size):
    height, width = image.shape
    patch_size = int(min(patch_size, height, width))
    if patch_size < 8:
        yield image
        return

    crop_h = (height // patch_size) * patch_size
    crop_w = (width // patch_size) * patch_size
    if crop_h < patch_size or crop_w < patch_size:
        yield image
        return

    image = image[:crop_h, :crop_w]
    for top in range(0, crop_h, patch_size):
        for left in range(0, crop_w, patch_size):
            yield image[top:top + patch_size, left:left + patch_size]


def _append_features(features, reference, scale, values, ref_values, scale_values):
    features.extend(values)
    reference.extend(ref_values)
    scale.extend(scale_values)


def _il_niqe_gabor_kernels():
    kernels = []
    for sigma, wavelength in ((2.0, 4.0), (4.0, 8.0)):
        for theta in (0.0, np.pi / 2.0):
            kernels.append(cv2.getGaborKernel(
                ksize=(15, 15),
                sigma=sigma,
                theta=theta,
                lambd=wavelength,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            ))
    return kernels


def _il_niqe_patch_features(patch, include_gabor=False):
    patch = np.nan_to_num(np.asarray(patch, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    patch = np.clip(patch, 0.0, 1.0)
    features, reference, scale = [], [], []

    mscn, _, _ = compute_image_mscn_transform(patch * 255.0)
    alpha, var = _safe_ggd_features(mscn)
    _append_features(features, reference, scale,
                     [alpha, var],
                     [2.0, 1.0],
                     [1.5, 1.0])
    for product in paired_product(mscn):
        alpha, mean_param, left_var, right_var = _safe_aggd_features(product)
        _append_features(features, reference, scale,
                         [alpha, mean_param, left_var, right_var],
                         [1.0, 0.0, 0.5, 0.5],
                         [1.5, 1.0, 1.0, 1.0])

    smoothed = cv2.GaussianBlur(patch, ksize=(0, 0), sigmaX=1.0)
    grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    for gradient in (grad_x, grad_y):
        alpha, var = _safe_ggd_features(gradient)
        _append_features(features, reference, scale,
                         [alpha, var],
                         [1.0, 0.02],
                         [2.0, 0.2])
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    shape, scale_param = _safe_weibull_features(grad_mag)
    _append_features(features, reference, scale,
                     [shape, scale_param],
                     [2.0, 0.05],
                     [2.0, 0.2])

    if include_gabor:
        for kernel in _il_niqe_gabor_kernels():
            response = cv2.filter2D(patch, cv2.CV_32F, kernel)
            alpha, var = _safe_ggd_features(response)
            shape, scale_param = _safe_weibull_features(response)
            _append_features(features, reference, scale,
                             [alpha, var, shape, scale_param],
                             [1.0, 0.02, 2.0, 0.05],
                             [2.0, 0.2, 2.0, 0.2])

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(reference, dtype=np.float32),
        np.asarray(scale, dtype=np.float32),
    )


def _il_niqe_local_distance(feature_matrix, reference, ref_scale):
    feature_matrix = np.nan_to_num(np.asarray(feature_matrix, dtype=np.float32),
                                   nan=0.0, posinf=0.0, neginf=0.0)
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix[None, :]
    sample_var = np.var(feature_matrix, axis=0) if feature_matrix.shape[0] > 1 else np.zeros_like(reference)
    cov_diag = 0.5 * (sample_var + ref_scale ** 2) + 1e-8
    distances = np.sqrt(np.mean(((feature_matrix - reference) ** 2) / cov_diag, axis=1))
    return float(np.mean(distances))


def _il_niqe_channel_score(channel_data, patch_size=84, include_gabor=False):
    patch_features = []
    reference = None
    ref_scale = None
    for patch in _iter_quality_patches(channel_data, patch_size):
        features, patch_reference, patch_scale = _il_niqe_patch_features(
            patch, include_gabor=include_gabor)
        patch_features.append(features)
        reference = patch_reference
        ref_scale = patch_scale

    return _il_niqe_local_distance(np.vstack(patch_features), reference, ref_scale)


def _il_niqe_spectral_opponent_score(input_img, patch_size=84):
    if input_img.shape[2] < 3:
        return None

    band_indices = [0, input_img.shape[2] // 2, input_img.shape[2] - 1]
    pseudo_rgb = np.clip(input_img[:, :, band_indices], 0.0, 1.0)
    log_rgb = np.log(pseudo_rgb + 1e-6)
    log_rgb = log_rgb - np.mean(log_rgb, axis=(0, 1), keepdims=True)
    r, g, b = log_rgb[:, :, 0], log_rgb[:, :, 1], log_rgb[:, :, 2]
    opponent = np.stack((
        (r + g + b) / np.sqrt(3.0),
        (r + g - 2.0 * b) / np.sqrt(6.0),
        (r - g) / np.sqrt(2.0),
    ), axis=2)

    patch_features = []
    reference = None
    ref_scale = None
    for channel in range(opponent.shape[2]):
        for patch in _iter_quality_patches(opponent[:, :, channel], patch_size):
            values = np.asarray([np.mean(patch), np.var(patch)], dtype=np.float32)
            patch_reference = np.asarray([0.0, 0.05], dtype=np.float32)
            patch_scale = np.asarray([1.0, 0.5], dtype=np.float32)
            patch_features.append(values)
            reference = patch_reference
            ref_scale = patch_scale

    return _il_niqe_local_distance(np.vstack(patch_features), reference, ref_scale)


def calculate_il_niqe(inputImgData):
    """
    Calculate a self-contained IL-NIQE-style no-reference score for HSI data.

    The implementation follows the Integrated Local NIQE idea by using
    patch-level NSS distances with enhanced MSCN, adjacent-product, gradient,
    compact directional-frequency, and spectral-opponent descriptors. It does
    not require an external pristine-model or PCA file; lower scores indicate
    statistics closer to the pristine natural-image proxy.
    """
    inputImgData = _quality_normalize(inputImgData)
    component_scores = []

    channel_scores = [
        _il_niqe_channel_score(inputImgData[:, :, c], include_gabor=False)
        for c in range(inputImgData.shape[2])
    ]
    component_scores.append(float(np.mean(channel_scores)))

    spectral_mean = np.mean(inputImgData, axis=2)
    component_scores.append(_il_niqe_channel_score(spectral_mean, include_gabor=True))

    opponent_score = _il_niqe_spectral_opponent_score(inputImgData)
    if opponent_score is not None:
        component_scores.append(opponent_score)

    return float(np.mean(component_scores))
