import datetime
import datetime
import json
import os

import cv2
import numpy as np
import torch
#
from biosppy.signals import bvp
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import affine_transform
from sklearn.model_selection import KFold
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.dataset_loader import dataset_loader
from loss import loss_fn
from models import get_ver_model
from pytorch_grad.utils.image import show_cam_on_image

# 2번 GPU만 사용하고 싶은 경우 예시(cuda:0에 지정)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

bpm_flag = False
K_Fold_flag = False
model_save_flag = False
# Define Kfold Cross Validator
if K_Fold_flag:
    kfold = KFold(n_splits=5, shuffle=True)

# wandb.init(project="SeqNet",entity="daeyeolkim")

now = datetime.datetime.now()
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

with open('params.json') as f:
    jsonObject = json.load(f)
    __PREPROCESSING__ = jsonObject.get("__PREPROCESSING__")
    __TIME__ = jsonObject.get("__TIME__")
    __MODEL_SUMMARY__ = jsonObject.get("__MODEL_SUMMARY__")
    options = jsonObject.get("options")
    params = jsonObject.get("params")
    hyper_params = jsonObject.get("hyper_params")
    model_params = jsonObject.get("model_params")
#
"""
TEST FOR LOAD
"""
dataset = dataset_loader(save_root_path=params["save_root_path"],
                         model_name=model_params["name"],
                         dataset_name=params["dataset_name"],
                         option="train")

train_loader = DataLoader(dataset, batch_size=params["train_batch_size"],
                          shuffle=params["train_shuffle"])
# ncols = 32
# nrows = 10
#
# # create the plots
# fig = plt.figure(figsize=(20,10))
# axes = []
#
# for r in range(nrows):
#     for c in range(ncols):
#         axes.append(fig.add_subplot(nrows,ncols,r*ncols+c+1))
#
# # remove the x and y ticks
# for ax in axes:
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.subplots_adjust(wspace=0.0,hspace=0.0)

video, label = next(iter(train_loader))
X_var = Variable(video.cpu(), requires_grad=True)
y_var = Variable(label.cpu())

for ver in range(10):
    # ver = 5
    # ver = 6
    if ver == 0:  # M(W@H)+M
        model_name = "APNET_M(W+H)+M"
    elif ver == 1:  # M(W+H)+M
        model_name = "APNET_M(WH)+M"
    elif ver == 2:  # MW+M
        model_name = "APNET_WM+M"
    elif ver == 3:  # MH+M
        model_name = "APNET_HM+M"
    elif ver == 4:  # M
        model_name = "APNET_NO"
    elif ver == 5:  # WM + HM
        model_name = "APNETWM+HM"
    elif ver == 6:  # WM
        model_name = "APNETWM"
    elif ver == 7:  # HM
        model_name = "APENTHM"
    elif ver == 8:  # H
        model_name = "APNETH"
    elif ver == 9:  # W
        model_name = "APNETW"

    fig = plt.figure(figsize=(40, 4))
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    # model = get_model(model_params["name"])
    model = get_ver_model(model_params["name"], ver)
    model_dict = torch.load(params["model_root_path"] + model_name)
    model.load_state_dict(model_dict)

    # optimizer = optimizer(model.parameters(), hyper_params["learning_rate"], hyper_params["optimizer"])
    # optimizer.zero_grad()
    model_name_param = dict(model.named_parameters())
    main_feature = model_name_param['sa_main.conv1.weight']
    bvp_feature = model_name_param['sa_bvp.conv1.weight']
    ptt_feature = model_name_param['sa_ptt.conv1.weight']

    layer = list(model.children())

    model_weights = []
    conv_layers = []

    output, m8, b8, p8 = model(X_var)
    criterion = loss_fn(hyper_params["loss_fn"])

    loss = criterion(output, y_var)
    print(ver)
    print(loss)
    loss.backward()

    images_grads = X_var.grad.data[0]
    abs_images_grads = images_grads.abs()
    saliency, _ = abs_images_grads.max(dim=0)
    saliency = saliency.cpu().detach().numpy()
    # saliency = saliency/np.max(saliency)
    vid = video[0].abs()
    vid, _ = vid.max(dim=0)
    vid = vid.cpu().detach().numpy()
    U = vid + 1.2 * saliency
    # U = saliency
    U -= U.mean()
    U /= U.std()
    U += np.abs(np.min(U))
    U = U / np.max(U)
    # U = saliency

    vid = torch.mean(video[0], dim=[0]).cpu().detach().numpy()
    for i in range(len(saliency)):
        # a_s = saliency[i] / np.max(saliency[i])
        a_s = saliency[i] * 255
        # a_s = np.exp2(a_s)
        # a_s = saliency[i].cpu().detach().numpy() * (1/np.min(saliency[i].cpu().detach().numpy()))
        # a_s = np.log(a_s)
        # a_s = np.log(a_s)
        ax = fig.add_subplot(2, 32, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        v_s = vid[i]
        # t = a_s *0.4 + v_s*255
        t = a_s
        # t = a_s * v_s
        # ax.imshow(t,'jet')
        ax.imshow(t)
        # plt.imshow(t, 'jet')
        # plt.show()

    for i in range(0, 1, 1):
        plt.title("ver" + str(ver))
        ax2 = fig.add_subplot(2, 1, 2)
        xticks = [i for i in range(0, 32, 1)]
        ax2.set_xticks(xticks)
        ax2.plot(torch.reshape(output, shape=(-1,)).cpu().detach().numpy()[i * 32:(i + 1) * 32], label="Inference")
        ax2.plot(torch.reshape(y_var, shape=(-1,))[i * 32:(i + 1) * 32], label="Target")
        ax2.legend()
        ax2.margins(x=1 / 64, y=0)
        ax2.grid(True, axis='x', linewidth=3, linestyle=":")

        # plt.cla()
    plt.show()

layers = []

images_grads = X_var.grad.data[1]
abs_images_grads = images_grads.abs()
# abs_images_grads = torch.permute(abs_images_grads,(1,2,3,0))
saliency, _ = abs_images_grads.max(dim=0)
saliency = saliency.cpu().detach().numpy()
# saliency = np.maximum(saliency, 0)
vid = video[1]
vid = torch.permute(vid, (1, 2, 3, 0))
# saliency = saliency/np.max(saliency)
# vid = video[1].abs()
# vid,_ = vid.max(dim=0)
vid = vid.cpu().detach().numpy()
# saliency = np.maximum(saliency, 0)
U = []
for v, s in zip(vid, saliency):
    s = np.maximum(s, 0)
    s /= np.max(s)
    U.append(show_cam_on_image(v, s, use_rgb=True))
# U = vid + 1.2*saliency
# U = saliency
# U -= U.mean()
# U /= U.std()
# U += np.abs(np.min(U))
# U = U / np.max(U)
# U = saliency
U = np.asarray(U)
min_val = U.min()
max_val = U.max()
n_x, n_y, n_z, _ = U.shape
colormap = plt.cm.jet

for i in range(n_x):
    U[i] = cv2.rotate(U[i], cv2.ROTATE_90_CLOCKWISE)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30, 60)
for i in range(n_y):
    y_cut = U[:, i, :]
    X, Z = np.mgrid[0:n_x, 0:n_z]
    Y = i * np.ones((n_x, n_z))
    # fig = plt.figure()
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colormap((y_cut - min_val) / (max_val - min_val)),
                    shade=False)

plt.show()

# Random test images.
# rs = np.random.RandomState(123)
# img = rs.randn(img_height, img_width)*0.1
# images = [img+(i+1) for i in range(nimages)]


#
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# img = ax.contour3D(X,Y,Z,c=video[0].cpu().numpy(),cmap=plt.hot())
# fig.colorbar(img)
# plt.show()


saliency = None
ct = 0
# for child in model.children():
#     ct+=1
#     print(ct,child)
#     if ct !=4:
#         for param in child.parameters():
#             param.requires_grad = False

# for name, child in model.named_children():
#     print(name)
#     for child_name,child_child in child.named_children():
#         print(child_name)
#
#
# for idx,m in enumerate(model.named_modules()):
#     print(idx, '->', m)

# model.eval()
# optimizer = optimizer(model.parameters(), hyper_params["learning_rate"], hyper_params["optimizer"])
# optimizer.zero_grad()

# loss.register_hook(lambda grad:print(grad))


plt.plot((output[0].cpu().detach().numpy()))
plt.show()

idx = 9

plt.cla()

ptt_img = ptt[5][0]
ptt_img = torch.permute(ptt_img, (1, 2, 3, 0))
ptt_img = torch.reshape(ptt_img, (128, 32 * 8, 3))
ptt_img = ptt_img.cpu().detach().numpy()
ptt_img -= np.mean(ptt_img)
ptt_img /= np.std(ptt_img)
# ptt_img += np.abs(np.min(ptt_img))
# ptt_img /= np.max(ptt_img)
plt.imshow(ptt_img, 'jet')
plt.show()
main_img, _ = main[6][0].max(dim=0)
main_img = main_img.cpu().detach().numpy()
for i in range(len(main_img)):
    main_img[i] -= np.mean(main_img[i])
    main_img[i] /= np.std(main_img[i])
    main_img[i] += np.abs(np.min(main_img[i]))
    main_img[i] /= np.max(main_img[i])

plt.imshow(main_img, 'jet')
plt.show()
plt.cla()
ptt_img, _ = ptt[idx][0].max(dim=0)
ptt_img = ptt_img.cpu().detach().numpy()
for i in range(len(ptt_img)):
    ptt_img[i] -= np.mean(ptt_img[i])
    ptt_img[i] /= np.std(ptt_img[i])
    ptt_img[i] += np.abs(np.min(ptt_img[i]))
    ptt_img[i] /= np.max(ptt_img[i])

plt.imshow(ptt_img, 'jet')
plt.show()
plt.cla()
bvp_img, _ = bvp[idx][0].max(dim=0)
bvp_img = bvp_img.cpu().detach().numpy()
for i in range(len(ptt_img)):
    bvp_img[i] -= np.mean(bvp_img[i])
    bvp_img[i] /= np.std(bvp_img[i])
    bvp_img[i] += np.abs(np.min(bvp_img[i]))
    bvp_img[i] /= np.max(bvp_img[i])

plt.imshow(bvp_img, 'jet')
plt.show()

z = np.linspace(0, 1, 128)  # y축
x = np.linspace(0, 1, 128)  # x축
y = np.linspace(0, 1, 32)  # T축

X, Y, Z = np.meshgrid(x, y, z)

#
images_grads = X_var.grad.data[1]
abs_images_grads = images_grads.abs()
saliency, _ = abs_images_grads.max(dim=0)
saliency = saliency.cpu().detach().numpy()
# saliency = saliency/np.max(saliency)
vid = video[1].abs()
vid, _ = vid.max(dim=0)
vid = vid.cpu().detach().numpy()
U = vid + 1.2 * saliency
# U = saliency
U -= U.mean()
U /= U.std()
U += np.abs(np.min(U))
U = U / np.max(U)
# U = saliency

min_val = U.min()
max_val = U.max()
n_x, n_y, n_z = U.shape
colormap = plt.cm.jet

for i in range(n_x):
    U[i] = cv2.rotate(U[i], cv2.ROTATE_90_CLOCKWISE)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30, 60)
for i in range(n_y):
    y_cut = U[:, i, :]
    X, Z = np.mgrid[0:n_x, 0:n_z]
    Y = i * np.ones((n_x, n_z))
    # fig = plt.figure()
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colormap((y_cut - min_val) / (max_val - min_val)),
                    shade=False)

plt.show()
# ax.set_title("y slice")


permuted_sal = torch.permute(abs_images_grads, (1, 2, 3, 0)).detach().cpu().numpy()
# vid = torch.mean(video[0],dim=[0])
video_sal = torch.permute(video[0], (1, 2, 3, 0)).cpu().numpy()
#
# video = torch.permute(video,(0,1,2)).cpu().detach().numpy()
# permuted_sal = np.mean(permuted_sal,axis=3)
# vid = torch.mean(video[0],dim=[0]).cpu().detach().numpy()

# saliency_transpose,_ = torch.permute(abs_images_grads,(0,2,1,3)).max(dim=0)
# saliency_transpose = saliency_transpose.cpu().detach().numpy()
# video_transpose = torch.permute(video[0],(2,1,3,0)).cpu().numpy()
# test_vid_trp = np.mean(video_transpose,axis=3)
# for i in range(len(saliency_transpose)):
#     a_s = saliency_transpose[i]/np.max(saliency_transpose[i])
#     test_vid_tmp = test_vid_trp[i] + 1.2*a_s
#     plt.imshow(test_vid_tmp,'jet')
#     plt.show()

saliency_transpose, _ = torch.permute(abs_images_grads, (0, 3, 1, 2)).max(dim=0)
saliency_transpose = saliency_transpose.cpu().detach().numpy()
video_transpose = torch.permute(video[0], (3, 1, 2, 0)).cpu().numpy()
test_vid_trp = np.mean(video_transpose, axis=3)
for i in range(len(saliency_transpose)):
    a_s = saliency_transpose[i] / np.max(saliency_transpose[i])
    test_vid_tmp = test_vid_trp[i] + 1.2 * a_s
    plt.imshow(test_vid_tmp, 'jet')
    plt.show()

# t = permuted_sal*1.2  + vid
# t = np.reshape(np.transpose(vid + permuted_sal*1.2,(1,0,2)),(128,-1))

# a_s = saliency[i].cpu().detach().numpy()/np.max(saliency[i].cpu().detach().numpy())
# t = a_s*1.2  + v_s

test_vid = np.mean(video, axis=3)
test_vid = np.transpose(test_vid, (2, 1, 0))
saliency = np.transpose(saliency, (2, 1, 0)).cpu().detach().numpy()
for i in range(len(saliency)):
    a_s = saliency[i] / np.max(saliency)
    test_vid[i] = test_vid[i] + 1.2 * a_s
    # for j in range(3):
    #     video[i,:,:,j] =video[i,:,:,j] + 1.2*a_s

# video = video + permuted_sal


# trp_test_vid = np.transpose(test_vid,(2,1,0))
plt.axis('off')
for i in range(len(test_vid[i])):
    plt.imshow(test_vid[i], 'jet')
    # plt.show()
    plt.savefig('./ver/' + params["dataset_name"] + str(i) + '.png', bbox_inches='tight', pad_inches=0)

nimages = 32
img_height, img_width = 128, 64
bg_val = 0  # Some flag value indicating the background.

interval = 32

y_factor = 0.5

images = []
# video = torch.permute(video[0],(1,2,3,0)).cpu().numpy()
for i in range(32):
    images.append(video[i][:, :64])
    # images.append(cv2.cvtColor(video[i],cv2.COLOR_BGR2BGRA))

stacked_height = int((1 + y_factor) * (img_height))
stacked_width = int(img_width + ((nimages - 1) * img_width / interval) * (1 / y_factor))
stacked_channel = 3
stacked = np.full((stacked_height, stacked_width, stacked_channel), bg_val).astype(np.float32)

# Affine transform matrix.
T = np.array([[1, -1 * y_factor],
              [0, 1]])

for i in range(nimages):
    # The first image will be right most and on the "bottom" of the stack.
    o = int((nimages - i - 1) * img_width / interval)
    out = np.empty((stacked_height, stacked_width, stacked_channel))
    for j in range(stacked_channel):
        out[:, :, j] = affine_transform(images[i][:, :, j], T, offset=[o, -o * 2], output_shape=stacked.shape[:2],
                                        cval=bg_val)
    stacked[out != bg_val] = out[out != bg_val]
#
# plt.imshow(stacked, cmap=plt.cm.jet)
plt.imshow(stacked, 'jet')
plt.show()
stacked_mean = np.mean(stacked, axis=2)
plt.imshow(stacked_mean, 'jet')
plt.show()
#
#


for i in range(len(permuted_sal)):
    # permuted_sal[i] = np.mean(permuted_sal[i],axis=2)
    permuted_sal[i] -= np.mean(permuted_sal[i])
    permuted_sal[i] /= np.std(permuted_sal[i])

    permuted_sal[i] = permuted_sal[i] / np.max(permuted_sal[i])

hil = np.reshape(np.transpose(vid * 0.4 + permuted_sal * 0.6, (1, 0, 2)), (128, -1))

vid = torch.mean(video[0], dim=[0]).cpu().detach().numpy()
for i in range(len(saliency)):
    a_s = saliency[i].cpu().detach().numpy() / np.max(saliency[i].cpu().detach().numpy())
    # a_s = np.exp2(a_s)
    # a_s = saliency[i].cpu().detach().numpy() * (1/np.min(saliency[i].cpu().detach().numpy()))
    # a_s = np.log(a_s)
    # a_s = np.log(a_s)
    v_s = vid[0]
    t = a_s * 1.2 + v_s
    # t = a_s * v_s
    plt.imshow(t, 'jet')
    plt.show()
    print("A")

print("A")
