import random
from glob import glob

import torch
from nets.UNet import UNet

import cv2
import face_recognition
import mediapipe as mp
import numpy as np

from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor

import matplotlib.pyplot as plt


def getFace(frame):
    face_locations = face_recognition.face_locations(frame, 1, model='hog')
    if len(face_locations) >= 1:
        (bottom, right, top, left) = face_locations[0]
    else:
        exit(0)

    y_range_ext = (top - bottom) * 0.2  # for forehead
    bottom = bottom - y_range_ext

    cnt_y = round(((top + bottom) / 2))
    cnt_x = round((right + left) / 2)
    bbox_half_size = round((top - bottom) * (1.5 / 2))

    face = np.take(frame, range(cnt_y - bbox_half_size, cnt_y + bbox_half_size), 0, mode='clip')
    face = np.take(face, range(cnt_x - bbox_half_size, cnt_x + bbox_half_size), 1, mode='clip')

    return face


def getSkin(frame, model, device):
    channel_first_img = np.transpose(frame, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    img_added_axis = np.expand_dims(channel_first_img, axis=0)
    input_tensor = torch.from_numpy(img_added_axis).float()
    input_tensor.to(device=device)

    preds = model(input_tensor)
    prediction = preds[0].cpu().detach().numpy()
    prediction = np.transpose(prediction, (1, 2, 0))  # (C, H, W) -> (H, W, C)

    prediction_mask = np.uint8((prediction > 0.95) * 255)
    skin = cv2.bitwise_and(frame, frame, mask=prediction_mask)

    # skin = face * prediction

    return skin


def getROI(frame):
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    results = face_mesh.process(frame)
    if len(results.multi_face_landmarks) >= 1:
        face_lm = results.multi_face_landmarks[0].landmark
    else:
        exit(0)

    lm_points = []
    ih, iw, ic = frame.shape
    for lm in face_lm:
        x, y = int(lm.x * iw), int(lm.y * ih)
        lm_points.append([x, y])

    if len(lm_points) > 0:
        lm_points_np = np.array(lm_points)
        lm_roi = make_specific_mask([2, 3, 20], lm_points_np)  # 2:Malar_left(왼뺨), 3:Malar_right(오른뺨), 20:Glabella(미간),
        roi = np.array(lm_roi, dtype=object)
    else:
        print(f'Error getROI() : Empty Face')
        exit()

    return roi


def getROI_RGB(frame, roi):
    h, w, c = frame.shape

    roi_RGB = []
    for mask in roi:
        view_mask = np.zeros((h, w, c), np.uint8)
        view_mask = cv2.fillConvexPoly(view_mask, mask.astype(int), color=(255, 255, 255))
        r = cv2.bitwise_and(frame, view_mask)
        r = r.reshape(-1, 3)
        r_rgb_idx = (r[:, 0] != 0)
        r_rgb = r[r_rgb_idx, :]
        roi_RGB.append(r_rgb)
    return roi_RGB


def getSkin_RGB(roi_RGB):
    skin_RGB_list = []
    for channel in range(3):
        channel_roi_mean_sum = 0.
        for roi_num in range(len(roi_RGB)):
            channel_roi_mean_sum += roi_RGB[roi_num][:, channel].mean()

        skin_RGB_list.append(channel_roi_mean_sum / len(roi_RGB))

    return np.array(skin_RGB_list, np.float32)


def make_specific_mask(bin_mask, dot):
    '''
    :param bin_mask:
    0 :  lower_cheek_left,          1 :  lower_cheek_right,          2 :  Malar_left
    3 :  Malar_right,               4 :  Marionette_Fold_left        5 :  Marionette_Fold_right
    6 :  chine                      7 :  Nasolabial_Fold_left        8 :  Nasolabial_Fold_right
    9 :  Lower_NasaL_Sidewall       10:  Upper_Lip_left              11:  Upper_Lip_right
    12:  Philtrum                   13:  Nasal_Tip                   14:  Lower_Nasal_Dorsum
    15: Lower_Nasal_Sidewall_left   16:  Lower_Nasal_Sidewall_right  17:  Mid_Nasal_Sidewall_left
    18: Mid_Nasal_Sidewall_right    19:  Upper_Nasal_Dorsum          20:  Glabella
    21: Lower_Lateral_Forehead_left 22: Lower_Lateral_Forehead_right 23: temporal_lobe_left
    24: temporal_lobe_right         25: eye_left                     26: eye_right
    27: Lower_Medial_Forehead       28: Upper_Lateral_Forehead_left  29: Upper_Lateral_Forehead_right
    30: Upper_Medial_Forehead
    :param dot:
    :return:
    '''
    mask_list = []
    for idx in bin_mask:
        if idx == 0:  # lower_cheek_left
            mask_list.append(np.array(
                [avg(dot[132], dot[123]), dot[132], dot[215], dot[172], dot[136],
                 dot[169], dot[210], avg(dot[212], dot[202]), dot[57],
                 avg(dot[61], dot[186]),
                 dot[92], dot[206], dot[205], avg(dot[50], dot[147])]
            ))
        elif idx == 1:
            mask_list.append(np.array(
                [avg(dot[361], dot[352]), dot[361], dot[435], dot[397], dot[365],
                 dot[394], dot[430], avg(dot[273], dot[287]), dot[287],
                 avg(dot[391], dot[410]),
                 dot[322], dot[426], dot[425], avg(dot[411], dot[280])]))
        elif idx == 2:
            mask_list.append(np.array(
                [avg(dot[116], dot[93]), dot[116], dot[117], dot[118], dot[100],
                 avg(dot[126], dot[142]),
                 avg(dot[209], dot[142]), dot[49], dot[203], dot[205], dot[123]]
            ))
        elif idx == 3:
            mask_list.append(
                np.array([avg(dot[345], dot[323]), dot[345], dot[346], dot[347], dot[329],
                          avg(dot[420], dot[371]),
                          avg(dot[371], dot[360]), dot[429], dot[423], dot[425], dot[352]]))
        elif idx == 4:
            mask_list.append(np.array(
                [dot[61], dot[43], dot[204], dot[32], dot[171], dot[148], dot[176],
                 dot[149],
                 dot[150],
                 dot[169], dot[210], avg(dot[212], dot[202]),
                 avg(dot[57], avg(dot[57], dot[43]))]))
        elif idx == 5:
            mask_list.append(np.array(
                [dot[291], dot[273], dot[424], dot[262], dot[396], dot[377], dot[400], dot[378],
                 dot[379],
                 dot[394], dot[430], avg(dot[432], dot[422]),
                 avg(dot[287], avg(dot[287], dot[273]))]))
        elif idx == 6:
            mask_list.append(np.array(
                [dot[204], avg(dot[106], dot[91]), avg(dot[182], dot[181]), avg(dot[83], dot[84]),
                 avg(dot[18], dot[17]), avg(dot[314], dot[313]), avg(dot[405], dot[406]),
                 avg(dot[321], dot[335]), dot[424], dot[262], dot[396], dot[377], dot[152],
                 dot[148],
                 dot[171], dot[32]]))
        elif idx == 7:
            mask_list.append(np.array(
                [dot[61], dot[92], avg(dot[206], dot[203]), dot[102], dot[48], dot[64], dot[98], dot[60],
                 dot[165]]))
        elif idx == 8:
            mask_list.append(np.array(
                [dot[291], dot[322], avg(dot[426], dot[423]), dot[331], dot[294], dot[327], dot[290], dot[391]]))
        elif idx == 9:
            mask_list.append(np.array(
                [dot[240], dot[64], dot[48], dot[131], avg(dot[134], avg(dot[134], dot[220])),
                 dot[45], avg(dot[4], avg(dot[4], dot[1])),
                 dot[275], avg(dot[363], avg(dot[363], dot[440])), dot[360], dot[278], dot[294],
                 dot[460], dot[305], dot[309], dot[438],
                 avg(dot[457], dot[275]), avg(dot[274], dot[275]), dot[274],
                 dot[458], dot[461], dot[462], dot[370],
                 dot[94], dot[141],
                 dot[242], dot[241], dot[238], dot[44],
                 avg(dot[45], dot[44]),
                 avg(dot[218], dot[45]), dot[218], dot[79], dot[75]]))
        elif idx == 10:
            mask_list.append(np.array(
                [dot[60], dot[165], dot[61], dot[40], dot[39], dot[37], dot[97], dot[99]]))
        elif idx == 11:
            mask_list.append(np.array(
                [dot[290], dot[391], dot[291], dot[270], dot[269], dot[267], dot[326],
                 dot[328]]))
        elif idx == 12:
            mask_list.append(np.array(
                [dot[2], avg(dot[242], dot[97]), avg(dot[37], avg(dot[242], dot[37])), dot[0],
                 avg(dot[267], avg(dot[267], dot[462])), avg(dot[462], dot[326])]))
        elif idx == 13:
            mask_list.append(np.array([dot[4], avg(dot[45], avg(dot[45], dot[51])), dot[134],
                                       dot[51], dot[5], dot[281], dot[363],
                                       avg(dot[275], avg(dot[275], dot[281]))]))
        elif idx == 14:
            mask_list.append(np.array(
                [dot[195], dot[248], dot[456], avg(dot[281], avg(dot[281], dot[248]))
                    , dot[5], avg(dot[51], avg(dot[51], dot[3])), dot[236], dot[3]]))
        elif idx == 15:
            mask_list.append(
                np.array([dot[236], avg(dot[236], dot[134]),
                          avg(dot[198], dot[131]), avg(dot[126], dot[209]),
                          avg(dot[217], dot[198])])
            )
        elif idx == 16:
            mask_list.append(
                np.array([dot[456], avg(dot[456], dot[363]),
                          avg(dot[420], dot[360]), avg(dot[355], dot[429]),
                          avg(dot[437], dot[420])])
            )
        elif idx == 17:
            mask_list.append(
                np.array([dot[196], avg(dot[114], dot[217]), dot[236]])
            )
        elif idx == 18:
            mask_list.append(
                np.array([dot[419], avg(dot[343], dot[437]), dot[456]])
            )
        elif idx == 19:
            mask_list.append(
                np.array(
                    [avg(dot[114], dot[196]), dot[197], avg(dot[343], dot[419]), dot[351],
                     dot[417],
                     dot[285], dot[8],
                     dot[55], dot[193], dot[122]])
            )
        elif idx == 20:
            mask_list.append(
                np.array(
                    [avg(dot[55], avg(dot[55], dot[8])), dot[8],
                     avg(dot[285], avg(dot[8], dot[285])),
                     dot[336],
                     avg(dot[337], avg(dot[337], dot[336])),
                     avg(dot[151], avg(dot[151], dot[9])),
                     avg(dot[108], avg(dot[108], dot[107])), dot[107]])
            )
        elif idx == 21:
            mask_list.append(
                np.array(
                    [avg(dot[108], avg(dot[109], dot[69])),
                     avg(dot[104], avg(dot[68], dot[104])),
                     dot[105], dot[107]])
            )
        elif idx == 22:
            mask_list.append(
                np.array(
                    [avg(dot[337], avg(dot[338], dot[299])),
                     avg(dot[333], avg(dot[298], dot[333])),
                     dot[334], dot[336]])
            )
        elif idx == 23:
            mask_list.append(
                np.array(
                    [avg(dot[54], dot[103]), avg(dot[104], dot[68]), avg(dot[63], dot[105]),
                     dot[70],
                     dot[156], dot[143], dot[116], dot[234], dot[127],
                     dot[162], dot[21]])
            )
        elif idx == 24:
            mask_list.append(
                np.array(
                    [avg(dot[332], dot[284]), avg(dot[298], dot[333]),
                     avg(dot[334], dot[293]),
                     dot[300], dot[383], dot[372], dot[345], dot[454], dot[356],
                     dot[389], dot[251]])
            )
        elif idx == 25:
            mask_list.append(
                np.array(
                    [dot[53], dot[46], dot[156], dot[143], dot[111], dot[117], dot[118],
                     dot[119],
                     dot[120], dot[121],
                     dot[128], dot[245], dot[193], avg(dot[221], dot[55]), avg(dot[222], avg(dot[222], dot[65])),
                     dot[52]])
            )
        elif idx == 26:
            mask_list.append(
                np.array(
                    [dot[283], dot[276], dot[383], dot[372], dot[340], dot[346], dot[347],
                     dot[348], dot[349], dot[350], dot[357], dot[465], dot[417], dot[441],
                     avg(dot[442], avg(dot[442], dot[295])), avg(dot[285], dot[441]), dot[282]]))
        elif idx == 27:
            mask_list.append(
                np.array(
                    [
                        avg(dot[108], avg(dot[108], dot[109])),
                        avg(dot[151], avg(dot[10], avg(dot[151], avg(dot[151], dot[10])))),
                        avg(dot[337], avg(dot[337], dot[338])),
                        avg(dot[337], avg(dot[338], dot[299])),
                        avg(dot[337], avg(dot[337], dot[336])),
                        avg(dot[151], avg(dot[151], dot[9])),
                        avg(dot[108], avg(dot[108], dot[107])),
                        avg(dot[108], avg(dot[109], dot[69]))
                    ]
                )
            )
        elif idx == 28:
            mask_list.append(
                np.array(
                    [dot[103], dot[67], avg(dot[109], avg(dot[109], dot[67])),
                     avg(dot[108], avg(dot[108], dot[69])), dot[69], dot[104]])
            )
        elif idx == 29:
            mask_list.append(
                np.array(
                    [dot[332], dot[297], avg(dot[338], avg(dot[338], dot[297])),
                     avg(dot[337], avg(dot[337], dot[299])), dot[299], dot[333]])
            )
        elif idx == 30:
            mask_list.append(
                np.array(
                    [
                        avg(dot[108], avg(dot[109], dot[69])),
                        avg(dot[108], avg(dot[108], dot[109])),
                        avg(dot[151], avg(dot[10], avg(dot[151], avg(dot[151], dot[10])))),
                        avg(dot[337], avg(dot[337], dot[338])),
                        avg(dot[337], avg(dot[338], dot[299])), dot[338], dot[10], dot[109]
                    ]
                ))

    return mask_list


def avg(a, b):
    return [(int)((x + y) / 2) for x, y in zip(a, b)]


def get_roi_skin_rgb(frame, image_size, skin_model, device):
    # getFace (frame -> face)
    face = getFace(frame)
    face = cv2.resize(face, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # getSkin (face -> skin)
    skin = getSkin(face, skin_model, device)

    # getROI (face -> roi)
    roi = getROI(face)

    # getROI_RGB (skin, roi -> roi_RGB)
    roi_RGB = getROI_RGB(skin, roi)

    # getSkin_RGB (roiRGB -> skin_RGB)
    skin_RGB = getSkin_RGB(roi_RGB)

    # getGRGB (skin_RGB -> GRGB)
    # GRGB = (skin_RGB[:, 1]/skin_RGB[:, 0]) + (skin_RGB[:, 1]/skin_RGB[:, 2])

    return {'face': face, 'skin': skin, 'roi': roi, 'roi_RGB': roi_RGB, 'skin_RGB': skin_RGB}


def plot_roi_skin_rgb(result_dict, show=False):
    roi_img = np.zeros((result_dict['skin'].shape[0], result_dict['skin'].shape[1], 3), np.uint8)
    for mask in result_dict['roi']:
        roi_img = cv2.fillConvexPoly(roi_img, mask.astype(int), color=(255, 255, 255))
    roi_img = cv2.bitwise_and(result_dict['skin'], roi_img)

    RGB_img = np.ones((result_dict['skin'].shape[0], result_dict['skin'].shape[1], 3), np.uint8)
    RGB_img = RGB_img * result_dict['skin_RGB'].round().astype(np.uint8)

    debug_img_up = np.hstack([result_dict['face'], result_dict['skin']])
    debug_img_down = np.hstack([roi_img, RGB_img])
    debug_img = np.vstack([debug_img_up, debug_img_down])

    plt.axis("off")
    plt.imshow(debug_img)
    plt.tight_layout()
    if show:
        plt.show()


def hex2rgb(hex_str):
    if hex_str[0] == '#':
        hex_str = hex_str.lstrip('#')
    return list(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def hex2rgb_list(arr_hex):
    return [hex2rgb(hex_str) for hex_str in arr_hex]


def dominant_colors(image, n_clusters=3):
    data = np.reshape(image, (-1, 3))  # shape > (W*H, C)
    data = data[np.all(data != 0, axis=1)]  # 0을 포함한 RGB 픽셀 제거
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, colors = cv2.kmeans(data, n_clusters, None, criteria, 10, flags)
    labels, counts = np.unique(labels, return_counts=True)

    order = (-counts).argsort()
    colors = colors[order]
    counts = counts[order]

    props = counts / counts.sum()

    return colors, props


def getSkinType(color, prop, categories):
    lab_labels = [convert_color(sRGBColor.new_from_rgb_hex(lbl), LabColor) for lbl in categories]
    lab_color = convert_color(sRGBColor(rgb_r=color[0], rgb_g=color[1], rgb_b=color[2], is_upscaled=True), LabColor)
    distances = [np.sum([delta_e_cie2000(lab_color, label) * prop]) for label in lab_labels]
    label_id = np.argmin(distances)
    distance: float = distances[label_id]
    category_hex = categories[label_id]

    return label_id, category_hex, distance


def classifySkinType(skin_result_dict, categories, n_dominant_colors=3, debug=False):
    colors, props = dominant_colors(skin_result_dict['skin'], n_clusters=n_dominant_colors)

    # K-dominant-colors from K-mean algorithm (RGB)
    hex_colors = ['#%02X%02X%02X' % tuple(np.around([r, g, b]).astype(int)) for r, g, b in colors]

    # Skin Color chart/label
    label_id, category_hex, _ = getSkinType(colors[0], props[0], categories)

    dominant_rgb = hex2rgb_list(hex_colors)
    # category_hex => RGB
    skin_type_rgb = hex2rgb(category_hex)

    result = {'dominant_rgb': dominant_rgb[0], 'skin_type_rgb': skin_type_rgb}

    debug_img = None
    if debug:
        color_bars = []
        color_w = skin_result_dict['face'].shape[1]
        total_height = 0
        for index, color in enumerate(colors):
            # original_img : resized original image
            color_h = round(skin_result_dict['face'].shape[0] * props[index])
            total_height += color_h
            bar = create_bar(color_h, color_w, color)
            color_bars.append(bar)

        padding_height = skin_result_dict['face'].shape[0] - total_height
        if padding_height > 0:
            padding = create_bar(padding_height, color_w, (255, 255, 255))
            color_bars.append(padding)
        color_bars = np.vstack(color_bars)[:skin_result_dict['face'].shape[0], :, :]

        label_bars = []
        label_h = color_w
        label_w = (color_w * 3) // len(categories)
        label_bgrs = []
        for label_hex in categories:
            hex_val = label_hex.lstrip('#')
            lr, lg, lb = [int(hex_val[i:i + 2], 16) for i in (0, 2, 4)]
            label_bgrs.append([lb, lg, lr])
            label_bar = create_bar(label_h, label_w, [lb, lg, lr])
            label_bars.append(label_bar)
        padding_width = (color_w * 3) - (label_w * len(categories))
        if padding_width > 0:
            padding = create_bar(label_h, padding_width, (255, 255, 255))
            label_bars.append(padding)
        label_bars = np.hstack(label_bars)
        label_bars = cv2.cvtColor(label_bars, cv2.COLOR_BGR2RGB)

        debug_img = np.hstack([skin_result_dict['face'], skin_result_dict['skin'], color_bars])
        debug_img = np.vstack([debug_img, label_bars])

        label_y1 = label_h + 1
        label_y2 = label_y1 + label_h - 4
        label_x1 = label_id * label_w + 1
        label_x2 = label_x1 + label_w - 2
        debug_img = draw_rects(debug_img, [label_x1, label_y1, label_x2, label_y2], thickness=1)

        color_y1 = 2
        color_y2 = color_y1 + round(skin_result_dict['face'].shape[0] * props[0]) - 2
        color_x1 = color_w * 2 + 2
        color_x2 = color_x1 + color_w - 4
        debug_img = draw_rects(debug_img, [color_x1, color_y1, color_x2, color_y2], thickness=1)

    return result, debug_img


def draw_rects(image, *rects, color=(255, 0, 0), thickness=2):
    image = image.copy()
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color, thickness)
    return image


def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    return bar


# https://symbiosisonlinepublishing.com/computer-science-technology/computerscience-information-technology06.php
categories_HFFD = ('#2D221E',
                   '#3C2E28',
                   '#4B3932',
                   '#5A453C',
                   '#695046',
                   '#785C50',
                   '#87675A',
                   '#967264',
                   '#A57E6E',
                   '#B48A78',
                   '#C39582',
                   '#D2A18C',
                   '#E1AC96',
                   '#F0B8A0',
                   '#FFC3AA',
                   '#FFCEB4',
                   '#FFDABE',
                   '#FFE5C8')

# https://mellab-inha.github.io/static/pdf/CorrelationBetweenLightAbsorba.pdf
categories_Fitzpatrick = ('#6E3B07',
                          '#96734E',
                          '#D7A306',
                          '#EACD79',
                          '#FBEED2',
                          '#FEFEFE')

# https://opencriticalcare.org/faq/how-is-skin-color-measured-for-the-purposes-of-oximeter-performance-validation/
categories_VonLuschan = ('#1B1C2E',
                         '#312529',
                         '#402015',
                         '#573229',
                         '#603121',
                         '#653020',
                         '#643116',
                         '#794D30',
                         '#8E583E',
                         '#9C6B43',
                         '#BC9760',
                         '#C7A464',
                         '#DEB877',
                         '#DFC17B',
                         '#E1C16A',
                         '#E3C467',
                         '#EBD985',
                         '#EBF19F',
                         '#F2E297',
                         '#F4D293',
                         '#EFE2AD',
                         '#F1E7C3',
                         '#FFF9E1',
                         '#FEF6E1',
                         '#FCF8ED',
                         '#FBFCF4',
                         '#F4F1EA',
                         '#F3EAE5',
                         '#FAF0EF',
                         '#FEF7E5',
                         '#FDF6E6',
                         '#FDFBE6',
                         '#ECEBE9',
                         '#F4F2F5')

# https://skintone.google/
categories_Monk = ('#f6ede4',
                   '#f3e7db',
                   '#f7ead0',
                   '#eadaba',
                   '#d7bd96',
                   '#a07e56',
                   '#825c43',
                   '#604134',
                   '#3a312a',
                   '#292420')

skinColorCharts = {'HFFD': categories_HFFD,
                   'Fitzpatrick': categories_Fitzpatrick,
                   'VonLuschan': categories_VonLuschan,
                   'Monk': categories_Monk}

if __name__ == "__main__":
    seed = 42
    random.seed = seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    # Example Video (UBFC-rPPG)
    data_path = '/home/jh/data/UBFC/'
    video_path = [x + '/vid.avi' for x in glob(data_path + 'subject*')]

    # Pretrained model for skin-segmentation from https://github.com/MRE-Lab-UMD/abd-skin-segmentation
    model_path = '/home/jh/PycharmProjects/rppg/rppg/models/UNet/unet_for_skin.pth'
    model = UNet(input_channels=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()

    cap = cv2.VideoCapture(video_path[0])
    ret, frame = cap.read()
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

    # frame => {'face', 'skin', 'roi', 'roi_RGB', 'skin_RGB'}
    result_dict = get_roi_skin_rgb(frame, image_size=128, skin_model=model, device=device)

    # Plot ROI-Skin-RGB debug image
    plot_roi_skin_rgb(result_dict, show=True)
    # plt.savefig(f'./test_dir/skin.png')
    # plt.show()

    # Skin Type Chart: Fitzpatrick, Monk, HFFD, VonLuschan
    categories = skinColorCharts['Fitzpatrick']

    # getSkinType from ROI-Skin-RGB
    category_id, category_hex, _ = getSkinType(result_dict['skin_RGB'], prop=1., categories=categories)
    skin_type_rgb = hex2rgb(category_hex)

    # getSkinType from Skin Image
    result, debug_img = classifySkinType(result_dict, categories=categories, debug=True)

    plt.axis("off")
    plt.imshow(debug_img)
    plt.tight_layout()
    # plt.savefig(f'./test_dir/s{s_num}.png')
    plt.show()
