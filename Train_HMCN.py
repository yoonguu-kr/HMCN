'''
Warning : this .py file only runs below models

    'unet_diff':
    'MSF_diff':
    'MR_diff':
    'MRD_diff':

'''

import glob
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
import re
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
from medpy.metric.binary import dc, jc, hd, hd95, asd, assd, precision, recall, sensitivity, specificity



from config import _C as cfg
from Functions import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit, Dataset_epoch_validation, createFolder, fileNameUpdate, setup_logger, cuda_check, init_env, save_img, save_flow, transform_unit_flow_to_flow, load_4D, imgnorm, best_dc, tr_val_test, dice_multi
from HMCN import model_diff_l1, model_diff_l2, model_diff_l3, \
    SpatialTransform_unit, SpatialTransformNearest_unit, smoothloss, \
    neg_Jdet_loss, NCC, multi_resolution_NCC, kl_loss, lamda_mse_loss, contrastive_loss, ProbabilisticModel

module_name = sys.modules[model_diff_l1.__module__].__name__
print(f'module_name : {module_name}')

lr = cfg.SOLVER.LEARNING_RATE
checkpoint = cfg.SOLVER.CHECKPOINT

hyp_loss_jacobin = cfg.SOLVER.hyp_loss_jacobin
hyp_loss_smooth = cfg.SOLVER.hyp_loss_smooth
hyp_loss_CL = cfg.SOLVER.hyp_loss_CL
hyp_loss_kl = cfg.SOLVER.hyp_loss_kl
freeze_step = cfg.SOLVER.FREEZE_STEP
load_model = cfg.SOLVER.LOAD_MODEL

gpu_num = cfg.MODEL.GPU_NUM
doubleF = cfg.MODEL.FDoubleConv
doubleB = cfg.MODEL.BDoubleConv

ch_start = cfg.MODEL.Start_Channel
ch_start_magnitude = cfg.MODEL.Ch_magnitude


datatype = cfg.DATASET.DATATYPE
iteration_lvl1 = cfg.DATASET.Iter
iteration_lvl2 = cfg.DATASET.Iter
iteration_lvl3 = cfg.DATASET.Iter
data_norm = cfg.DATALOADER.NORM
batch = cfg.DATALOADER.BATCH_SIZE
num_worker = 4


print(f"cfg.DATASET.DATATYPE : {cfg.DATASET.DATATYPE}")
# to divide the Train, Valid Test & image label, mask
if cfg.DATASET.DATATYPE == 'Brain_OASIS':
    names = sorted(glob.glob(cfg.DATASET.DATA_PATH_IMGS))
    data_path_labels35 = cfg.DATASET.DATA_PATH_LABELS
    data_path_labels4 = data_path_labels35.replace("seg35", f"seg4")
    labels4 = sorted(glob.glob(data_path_labels4))
    labels35 = sorted(glob.glob(data_path_labels35))
    # labels = sorted(glob.glob(cfg.DATASET.DATA_PATH_LABELS))
    names_tr = names.copy()[:-82]
    names_val = names.copy()[-82:-41]
    names_test = names.copy()[-41:]

    labels_tr35 = labels35.copy()[:-82]
    labels_val35 = labels35.copy()[-82:-41]
    labels_test35 = labels35.copy()[-41:]

    labels_tr4 = labels4.copy()[:-82]
    labels_val4 = labels4.copy()[-82:-41]
    labels_test4 = labels4.copy()[-41:]


    # dataset에 Test용이 따로 있음
    ori_imgshape = (160, 192, 224)
    imgshape = (144, 160, 192)

    num_seg_class = 35
    num_classes = range(1, num_seg_class)
    ori_imgshape = (160, 192, 224)
    imgshape = (144, 160, 192)
    imgshape_2 = (144 // 2, 160 // 2, 192 // 2)
    imgshape_4 = (144 // 4, 160 // 4, 192 // 4)



# Create folders & Check the new file version that this file make
dir_Result = os.path.join(os.getcwd(), 'Result')
createFolder(dir_Result)


dir_datatype = os.path.join(dir_Result, datatype)
createFolder(dir_datatype)




def dice(im1, atlas, labels=None):
    if labels is None:
        unique_class = np.unique(atlas)
    else:
        unique_class = labels.copy()
    dice = 0
    num_count = 0
    eps = 1e-7
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        # print(sub_dice)
    # print(num_count, len(unique_class)-1)
    return dice / (num_count + eps)


def train_lvl1(dir_save, range_flow, load_model=False):
    print("Training diff lvl1...")

    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')

    model_lvl1 = model_diff_l1(2, ch_start, ch_start_magnitude, 3, is_train=True, imgshape=imgshape_4,
                               range_flow=range_flow, doubleF=doubleF, doubleB=doubleB, batchsize=batch).to(device)
    loss_similarity = NCC(win=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss
    loss_cl = contrastive_loss(batch_size=batch)

    # transform = SpatialTransform_unit().to(device)
    #
    # for param in transform.parameters():
    #     param.requires_grad = False
    #     param.volatile = True


    grid_4 = generate_grid(imgshape_4)
    grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()

    optimizer = torch.optim.Adam(model_lvl1.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    lossall = np.zeros((5, iteration_lvl1 + 1))

    training_generator = Data.DataLoader(Dataset_epoch(names_tr, norm=data_norm), batch_size=1,
                                         shuffle=True, num_workers=num_worker)
    step = 0
    # load_model = False
    # if load_model is True:
    #     model_path = os.path.join(dir_save, "")
    #     print("Loading weight: ", model_path)
    #     step = 3000
    #     model.load_state_dict(torch.load(model_path))
    #     temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
    #     lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl1:
        for X, Y in training_generator:
            X = X.to(device).float()
            Y = Y.to(device).float()
            # print("X.shape : ", X.shape) #torch.Size([1, 1, 160, 192, 224])

            X = F.interpolate(X, size=imgshape, mode='trilinear') #torch.Size([1, 1, 144, 160, 192])
            Y = F.interpolate(Y, size=imgshape, mode='trilinear')
            # print("X.shape after interpolate: ", X.shape)

            # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0  #original L1
            # F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y) #original L1
            F_X_Y, X_Y, Y_4x, F_xy, _, f_x,f_y = model_lvl1(X, Y)
            print(f'f_x.shape : {f_x.shape}')
            f_x = F.normalize(f_x, dim=1)
            f_y = F.normalize(f_y, dim=1)
            loss_CL = loss_cl(f_x, f_y)


            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            # reg2 - use velocity
            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            loss = loss_multiNCC + hyp_loss_jacobin * loss_Jacobian + hyp_loss_smooth * loss_regulation + hyp_loss_CL * loss_CL
            # loss = loss_multiNCC + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(), loss_CL.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}" - CL - "{5:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(), loss_CL.item()))
            sys.stdout.flush()

            logger_train.info(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}" - CL - "{5:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(), loss_CL.item()))

            # with lr 1e-3 + with bias
            if (step % checkpoint == 0):
                dir_modelname = os.path.join(dir_save, "_l1_" + str(step) + '.pth')
                torch.save(model_lvl1.state_dict(), dir_modelname)
                dir_npy1 = os.path.join(dir_save, 'loss_l1_' + str(step) + '.npy')
                np.save(dir_npy1, lossall)
            step += 1

            if step > iteration_lvl1:
                break
        # print("one epoch pass")
    # dir_npy1 = os.path.join(dir_save, 'loss_' + model_name + "l1.npy")
    dir_npy1 = os.path.join(dir_save, 'loss_l1_' + str(step) + '.npy')
    np.save(dir_npy1, lossall)


def train_lvl2(dir_save, range_flow, load_model=False):
    print("Training diff lvl2...")

    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')

    model_lvl1 = model_diff_l1(2, ch_start, ch_start_magnitude, 3, is_train=True, imgshape=imgshape_4,
                               range_flow=range_flow, doubleF=doubleF, doubleB=doubleB, batchsize=batch).to(device)

    model = model_diff_l2(2, ch_start, ch_start_magnitude, 3, is_train=True, imgshape=imgshape_2,range_flow=range_flow, model_lvl1=model_lvl1,
                          doubleF=doubleF, doubleB=doubleB,batchsize=batch).to(device)


    # model_path = "../Model/Stage/LDR_LPBA_NCC_1_1_stagelvl1_1500.pth"

    # find max epoch model
    files_model = [file for file in os.listdir(dir_save) if 'pth' in file]
    file_l1_model = [file for file in files_model if 'l1' in file]
    num_max = max([int(re.findall(r'\d+', number)[-1]) for number in file_l1_model])

    [model_max] = [file for file in file_l1_model if str(num_max) in file]
    print("model_max : ", model_max)
    model_path = os.path.join(dir_save, model_max)
    model_lvl1.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl1...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl1.parameters():
        param.requires_grad = False

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss
    loss_cl = contrastive_loss(batch_size=batch)

    # transform = SpatialTransform_unit().to(device)
    #
    # for param in transform.parameters():
    #     param.requires_grad = False
    #     param.volatile = True

    grid_2 = generate_grid(imgshape_2)
    grid_2 = torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lossall = np.zeros((5, iteration_lvl2 + 1))

    training_generator = Data.DataLoader(Dataset_epoch(names_tr, norm=data_norm), batch_size=1,
                                         shuffle=True, num_workers=num_worker)
    step = 0
    # load_model = False
    # if load_model is True:
    #     model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
    #     print("Loading weight: ", model_path)
    #     step = 3000
    #     model.load_state_dict(torch.load(model_path))
    #     temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
    #     lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl2:
        for X, Y in training_generator:

            X = X.to(device).float()
            Y = Y.to(device).float()

            X = F.interpolate(X, size=imgshape, mode='trilinear')
            Y = F.interpolate(Y, size=imgshape, mode='trilinear')

            # output_disp_e0, warpped_inputx_lvl1_out, y_down, compose_field_e0_lvl1v, lvl1_v, e0

            F_X_Y, X_Y, Y_2x, F_xy, F_xy_lvl1, _, f_x, f_y = model(X, Y)

            f_x = F.normalize(f_x, dim=1)
            f_y = F.normalize(f_y, dim=1)
            loss_CL = loss_cl(f_x, f_y)


            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_2x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            # reg2 - use velocity
            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            loss = loss_multiNCC + hyp_loss_jacobin * loss_Jacobian + hyp_loss_smooth * loss_regulation + hyp_loss_CL * loss_CL

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(), loss_CL.item()])
            sys.stdout.write(
                "\r" + ' lv2 : step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}" -CL "{5:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(), loss_CL.item()))
            sys.stdout.flush()

            logger_train.info(
                "\r" + ' lv2 : step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}" -CL "{5:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(), loss_CL.item()))

            # with lr 1e-3 + with bias
            if (step % checkpoint == 0):
                dir_modelname = os.path.join(dir_save, "_l2_" + str(step) + '.pth')
                # modelname = dir_save + '/' + model_name + "stagelvl2_" + str(step) + '.pth'
                torch.save(model.state_dict(), dir_modelname)
                dir_npy1 = os.path.join(dir_save, 'loss_l2_' + str(step) + '.npy')
                np.save(dir_npy1, lossall)

            if step == freeze_step:
                model.unfreeze_modellvl1()

            step += 1

            if step > iteration_lvl2:
                break
        print("one epoch pass")
    dir_npy1 = os.path.join(dir_save, 'loss_l2_' + str(step) + '.npy')
    np.save(dir_npy1, lossall)


def train_lvl3(dir_save, range_flow, load_model=False):
    print("Training diff lvl3...")

    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')

    model_lvl1 = model_diff_l1(2, ch_start, ch_start_magnitude, 3, is_train=True, imgshape=imgshape_4,
                               range_flow=range_flow, doubleF=doubleF, doubleB=doubleB, batchsize=batch).to(device)

    model_lvl2 = model_diff_l2(2, ch_start, ch_start_magnitude, 3, is_train=True, imgshape=imgshape_2,
                               range_flow=range_flow,model_lvl1=model_lvl1,
                               doubleF=doubleF,doubleB=doubleB, batchsize=batch).to(device)

    model = model_diff_l3(2, ch_start, ch_start_magnitude, 3, is_train=True, imgshape=imgshape,
                          range_flow=range_flow, model_lvl2=model_lvl2, doubleF=doubleF, doubleB=doubleB,batchsize=batch).to(device)

    logger_config.info("model_lvl1")
    logger_config.info(model_lvl1)
    logger_config.info("\n\n")
    logger_config.info("model_lvl2")
    logger_config.info(model_lvl2)
    logger_config.info("\n\n")
    logger_config.info("model")
    logger_config.info(model)
    logger_config.info("\n\n\n\n\n\n\n")

    # model_path = sorted(glob.glob("../Model/Stage/" + model_name + "stagelvl2_?????.pth"))[-1]
    # model_lvl2.load_state_dict(torch.load(model_path))

    # find max epoch model
    files_model = [file for file in os.listdir(dir_save) if 'pth' in file]
    file_l2_model = [file for file in files_model if 'l2' in file]
    num_max = max([int(re.findall(r'\d+', number)[-1]) for number in file_l2_model])

    [model_max] = [file for file in file_l2_model if str(num_max) in file]
    print("model_max : ", model_max)
    model_path = os.path.join(dir_save, model_max)
    model_lvl2.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False

    loss_similarity = multi_resolution_NCC(win=7, scale=3)

    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss
    loss_cl = contrastive_loss(batch_size=batch)

    # transform = SpatialTransform_unit().to(device)
    transform_nearest = SpatialTransformNearest_unit().to(device)
    #
    # for param in transform.parameters():
    #     param.requires_grad = False
    #     param.volatile = True

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    grid_unit = generate_grid_unit(ori_imgshape)
    grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # dir_save = os.path.join(dir_save, "Stage")
    # if not os.path.isdir(dir_save):
    #     os.mkdir(dir_save)

    lossall = np.zeros((5, iteration_lvl3 + 1))

    training_generator = Data.DataLoader(Dataset_epoch(names_tr, norm=data_norm), batch_size=1,
                                         shuffle=True, num_workers=num_worker)

    best_dice = best_dc()
    step = 0

    # load_model = False # to fix later
    # if load_model is True:
    #     model_path = "../Model/LDR_OASIS_NCC_unit_add_reg_3_anti_1_stagelvl3_10000.pth"
    #     print("Loading weight: ", model_path)
    #     step = 10000
    #     model.load_state_dict(torch.load(model_path))
    #     temp_lossall = np.load("../Model/lossLDR_OASIS_NCC_unit_add_reg_3_anti_1_stagelvl3_10000.npy")
    #     lossall[:, 0:10000] = temp_lossall[:, 0:10000]

    while step <= iteration_lvl3:
        for X, Y in training_generator:

            X = X.to(device).float()
            Y = Y.to(device).float()
            X = F.interpolate(X, size=imgshape, mode='trilinear')
            Y = F.interpolate(Y, size=imgshape, mode='trilinear')
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ , f_x, f_y= model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            f_x = F.normalize(f_x, dim=1)
            f_y = F.normalize(f_y, dim=1)
            loss_CL = loss_cl(f_x, f_y)



            # reg2 - use velocity
            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            loss = loss_multiNCC + hyp_loss_jacobin * loss_Jacobian + hyp_loss_smooth * loss_regulation + hyp_loss_CL * loss_CL

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(), loss_CL.item()])
            sys.stdout.write(
                "\r" + 'lv3 step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}" -CL "{5:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(), loss_CL.item()))
            sys.stdout.flush()

            logger_train.info(
                "\r" + ' lv3 : step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}" -CL "{5:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(), loss_CL.item()))

            if (step % checkpoint == 0):
                dir_modelname = os.path.join(dir_save, "_l3_" + str(step) + '.pth')
                # modelname = dir_save + '/' + model_name + "stagelvl2_" + str(step) + '.pth'
                torch.save(model.state_dict(), dir_modelname)

                dir_npy1 = os.path.join(dir_save, 'loss_l3_' + str(step) + '.npy')
                np.save(dir_npy1, lossall)

            if step == freeze_step:
                model.unfreeze_modellvl2()

            step += 1

            if step > iteration_lvl3:
                break
        print("one epoch pass")
        logger_eval.info("range_flow : {} Best Dice : {} Epoch : {} \n".format(range_flow, best_dice.get_best_dc()[0],
                                                                               best_dice.get_best_dc()[1]))
        dir_npy1 = os.path.join(dir_save, 'loss_l3_' + str(step) + '.npy')
        np.save(dir_npy1, lossall)

def test(dir_save,range_flow, model_name_front, model_name_back, doubleF, doubleB, max_only=False, brief = True, num_brief = 10, num_save=10):
    # dir_save= dir_epoch
    # reg_input1 = i
    # model_name_front = cfg.MODEL.NAME_FRONT
    # model_name_back = cfg.MODEL.NAME_BACK
    # doubleF = doubleF
    # doubleB = doubleB

    '''
    BEWARE!!!!!
    those below is string!!!!

    :param model: model name
    :param multi_dataset:
    :param epoch:
    :return:
    '''
    Log_test_total = f'Log_test_max_only_{max_only}_brief_{brief}_total.log'
    logger_test_total = setup_logger('test_total', dir_save, filename=Log_test_total)
    files_model = [pth_file for pth_file in os.listdir(dir_save) if 'pth' in pth_file]
    file_l3_model = [file for file in files_model if 'l3' in file]
    # these 2 code below is just for verification
    if max_only == True:
        num_max = max([int(re.findall(r'\d+', number)[-1]) for number in file_l3_model])
        file_l3_model = [file for file in file_l3_model if str(num_max) in file]

    logger_test_total.info(
        f'\tepoch \tfilel3 '
        f'\t\tstr_mean \t\tstr_std \t'
        f'\tdice_nanmean_list.mean() \tdice_nanmean_list.std() \t\tdice_sum_list.mean() \tdice_sum_list.std()'
        f'\t\tavg_dc.mean() \tavg_dc.std() \tavg_hd.mean() \tavg_hd.std() \tavg_hd95.mean()  \tavg_hd95.std()'
        f'\tavg_prc.mean() \tavg_prc.std() \tavg_rcl.mean() \tavg_rcl.std() \tavg_sensi.mean() \tavg_sensi.std()'
        f'\tavg_speci.mean() \tavg_speci.std() \tavg_asd.mean() \tavg_asd.std() '
        f'\tavg_assd.mean() \tavg_assd.std() \tavg_jc.mean() \tavg_jc.std()')

    for filel3 in file_l3_model:
        epoch = re.findall(r'\d+', filel3)[-1]
        folder_epoch = os.path.join(dir_save, 'brief_' + str(brief), 'epoch' + epoch)
        createFolder(folder_epoch)
        print(f'file_ep : {filel3}')
        Log_test = f'Log_test_max_only_{max_only}_brief_{brief}.log'
        logger_test = setup_logger('test', folder_epoch, filename=Log_test)
        logger_test.info(
            f'\tcur_iter \tfilel3 '
            f'\t\tstr_mean \t\tstr_std \t'
            f'\tdice_nanmean_list.mean() \tdice_nanmean_list.std() \t\tdice_sum_list.mean() \tdice_sum_list.std()'
            f'\t\tavg_dc.mean() \tavg_dc.std() \tavg_hd.mean() \tavg_hd.std() \tavg_hd95.mean()  \tavg_hd95.std()'
            f'\tavg_prc.mean() \tavg_prc.std() \tavg_rcl.mean() \tavg_rcl.std() \tavg_sensi.mean() \tavg_sensi.std()'
            f'\tavg_speci.mean() \tavg_speci.std() \tavg_asd.mean() \tavg_asd.std() '
            f'\tavg_assd.mean() \tavg_assd.std() \tavg_jc.mean() \tavg_jc.std()')


        init_env(str(gpu_num))
        device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)  # change allocation of current GPU

        model_lvl1 = model_diff_l1(2, ch_start, ch_start_magnitude, 3, is_train=True, imgshape=imgshape_4,
                                   range_flow=range_flow, doubleF=doubleF, doubleB=doubleB, batchsize=batch).to(device)

        model_lvl2 = model_diff_l2(2, ch_start, ch_start_magnitude, 3, is_train=True, imgshape=imgshape_2,
                                   range_flow=range_flow,model_lvl1=model_lvl1,doubleF=doubleF,doubleB=doubleB, batchsize=batch).to(device)

        model = model_diff_l3(2, ch_start, ch_start_magnitude, 3, is_train=False, imgshape=imgshape,
                              range_flow=range_flow, model_lvl2=model_lvl2, doubleF=doubleF, doubleB=doubleB, batchsize=batch).to(device)

        transform_nearest = SpatialTransformNearest_unit().to(device)
        transform = SpatialTransform_unit().to(device)
        model_path = os.path.join(dir_save, filel3)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        transform.eval()

        # grid_unit = generate_grid_unit(ori_imgshape)
        # grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).cuda().float()

        grid = generate_grid_unit(ori_imgshape)
        grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()
        # print(f'grid.shape : {grid.shape}')


        valid_generator = Data.DataLoader(Dataset_epoch_validation(names_test, labels_test4, norm=data_norm), batch_size=1,
                                          shuffle=False, num_workers=num_worker)


        dsc_list = []
        dice_nanmean_list = []
        dice_sum_list = []

        avg_dc = []
        avg_hd = []
        avg_hd95 = []
        avg_prc = []
        avg_rcl = []
        avg_sensi = []
        avg_speci = []
        avg_asd = []
        avg_assd = []
        avg_jc = []

        with torch.no_grad():
            for i, data in enumerate(valid_generator):
                if brief:
                    if i > num_brief:
                        break
                val_X, val_Y, val_X_label, val_Y_label = data[0].to(device), data[1].to(device), data[2].to(
                    device), data[3].to(device)

                val_X_ip = F.interpolate(val_X, size=imgshape, mode='trilinear')
                val_Y_ip = F.interpolate(val_Y, size=imgshape, mode='trilinear')
                # val_X_label = F.interpolate(val_X_label, size=imgshape, mode='trilinear')
                # val_Y_label = F.interpolate(val_Y_label, size=imgshape, mode='trilinear')

                # normalize image to [0, 1]
                norm = data_norm
                if norm:
                    val_X_ip = imgnorm(val_X_ip)
                    val_Y = imgnorm(val_Y)
                    # print(f'fixed_img.shape : {fixed_img.shape}')

                val_X_cpu = val_X.data.cpu().numpy()[0, 0, :, :, :]
                val_Y_cpu = val_Y.data.cpu().numpy()[0, 0, :, :, :]
                val_X_label_cpu = val_X_label.data.cpu().numpy()[0, 0, :, :, :]
                val_Y_label_cpu = val_Y_label.data.cpu().numpy()[0, 0, :, :, :]

                F_X_Y = model(val_X_ip, val_Y_ip)
                F_X_Y = F.interpolate(F_X_Y, size=ori_imgshape, mode='trilinear', align_corners=True)
                # print(f'val_X.shape : {val_X.shape}')
                # print(f'F_X_Y.shape : {F_X_Y.shape}')
                val_warpped_cpu = transform(val_X, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]


                F_X_Y_cpu = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
                # F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_model_output_cpu)


                val_warpped_label = transform_nearest(val_X_label, F_X_Y.permute(0, 2, 3, 4, 1), grid)
                val_warpped_label_cpu = val_warpped_label.data.cpu().numpy()[0, 0, :, :, :]


                # val_warpped, val_deform, _, _, _, _ = model(val_X, val_Y)
                # val_warpped_label = spatial_transform(val_X_label, val_deform, mode="nearest")
                # val_X_cpu = val_X.data.cpu().numpy()[0, 0, :, :, :]
                # val_Y_cpu = val_Y.data.cpu().numpy()[0, 0, :, :, :]
                # val_X_label_cpu = val_X_label.data.cpu().numpy()[0, 0, :, :, :]
                # val_Y_label_cpu = val_Y_label.data.cpu().numpy()[0, 0, :, :, :]
                #
                # val_warpped_cpu = val_warpped.data.cpu().numpy()[0, 0, :, :, :]
                # val_deform_cpu = val_deform.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
                # val_warpped_label_cpu = val_warpped_label.data.cpu().numpy()[0, 0, :, :, :]

                if i <= num_save:
                    dir_fix = os.path.join(folder_epoch, f'{i}_fixed.nii.gz')
                    save_img(val_Y_cpu, dir_fix)

                    dir_fix_label = os.path.join(folder_epoch, f'{i}_fixed_label.nii.gz')
                    save_img(val_Y_label_cpu, dir_fix_label)

                    dir_mov = os.path.join(folder_epoch, f'{i}_moving.nii.gz')
                    save_img(val_X_cpu, dir_mov)

                    dir_mov_label = os.path.join(folder_epoch, f'{i}_moving_label.nii.gz')
                    save_img(val_X_label_cpu, dir_mov_label)

                    dir_vectorfield = os.path.join(folder_epoch, f'{i}_vector_field.nii.gz')
                    save_flow(F_X_Y_cpu, dir_vectorfield)

                    dir_warppedMov = os.path.join(folder_epoch, f'{i}_warped_moving.nii.gz')
                    save_img(val_warpped_cpu, dir_warppedMov)

                    dir_warppedlabel = os.path.join(folder_epoch, f'{i}_warped_label.nii.gz')
                    save_img(val_warpped_label_cpu, dir_warppedlabel)
                dsc, volume = dice_multi(val_warpped_label.data.int().cpu().numpy(),
                                         val_Y_label.cpu().int().numpy())
                dsc_list.append(dsc)

                dice_ = np.nanmean(dsc)
                dice_nanmean_list.append(dice_)
                dice_sum = dice(val_warpped_label.data.int().cpu().numpy()[0, 0, :, :, :],
                                val_Y_label.cpu().int().numpy()[0, 0, :, :, :])
                dice_sum_list.append(dice_sum)


                # a = 1.0 / 100
                a = 1.0
                spacing = [a, a, a]
                avg_dc.append(dc(val_warpped_label_cpu, val_Y_label_cpu))
                avg_prc.append(precision(val_warpped_label_cpu, val_Y_label_cpu))
                avg_rcl.append(recall(val_warpped_label_cpu, val_Y_label_cpu))
                avg_sensi.append(sensitivity(val_warpped_label_cpu, val_Y_label_cpu))
                avg_speci.append(specificity(val_warpped_label_cpu, val_Y_label_cpu))

                # print(f'val_warpped_cpu.shape : {val_warpped_cpu.shape}')
                # print(f'val_Y_cpu.shape : {val_Y_cpu.shape}')
                if np.count_nonzero(val_warpped_cpu) == 0 or np.count_nonzero(val_Y_cpu) == 0:
                    avg_hd.append(np.nan)  # 혹은 0 또는 다른 placeholder
                    avg_hd95.append(np.nan)  # 혹은 0 또는 다른 placeholder
                    avg_asd.append(np.nan)  # 혹은 0 또는 다른 placeholder
                    avg_assd.append(np.nan)  # 혹은 0 또는 다른 placeholder
                    avg_jc.append(np.nan)  # 혹은 0 또는 다른 placeholder
                else:
                    avg_hd.append(hd(result=val_warpped_cpu, reference=val_Y_cpu, voxelspacing=spacing))
                    avg_hd95.append(
                        hd95(result=val_warpped_cpu, reference=val_Y_cpu, voxelspacing=spacing))
                    avg_asd.append(
                        asd(result=val_warpped_cpu, reference=val_Y_cpu, voxelspacing=spacing))
                    avg_assd.append(assd(result=val_warpped_cpu, reference=val_Y_cpu, voxelspacing=spacing))
                    avg_jc.append(jc(val_warpped_cpu, val_Y_cpu))

            avg_dc = np.array(avg_dc)
            avg_hd = np.array(avg_hd)
            avg_hd95 = np.array(avg_hd95)
            avg_prc = np.array(avg_prc)
            avg_rcl = np.array(avg_rcl)
            avg_sensi = np.array(avg_sensi)
            avg_speci = np.array(avg_speci)
            avg_asd = np.array(avg_asd)
            avg_assd = np.array(avg_assd)
            avg_jc = np.array(avg_jc)

            dsc_list = np.array(dsc_list)
            dice_nanmean_list = np.array(dice_nanmean_list)
            dice_sum_list = np.array(dice_sum_list)

            np.save(os.path.join(folder_epoch, 'avg_dc.npy'), avg_dc)
            np.save(os.path.join(folder_epoch, 'avg_hd.npy'), avg_hd)
            np.save(os.path.join(folder_epoch, 'avg_hd95.npy'), avg_hd95)
            np.save(os.path.join(folder_epoch, 'avg_prc.npy'), avg_prc)
            np.save(os.path.join(folder_epoch, 'avg_rcl.npy'), avg_rcl)
            np.save(os.path.join(folder_epoch, 'avg_sensi.npy'), avg_sensi)
            np.save(os.path.join(folder_epoch, 'avg_speci.npy'), avg_speci)
            np.save(os.path.join(folder_epoch, 'avg_asd.npy'), avg_asd)
            np.save(os.path.join(folder_epoch, 'avg_assd.npy'), avg_assd)
            np.save(os.path.join(folder_epoch, 'avg_jc.npy'), avg_jc)
            np.save(os.path.join(folder_epoch, 'dsc_list.npy'), dsc_list)
            np.save(os.path.join(folder_epoch, 'dice_nanmean_list.npy'), dice_nanmean_list)
            np.save(os.path.join(folder_epoch, 'dice_sum_list.npy'), dice_sum_list)

            str_mean = ''
            for i, val in enumerate(dsc_list.sum(axis=0)):
                str_mean += '\t{:.5f}'.format(val)

            str_std = ''
            for i, val in enumerate(dsc_list.std(axis=0)):
                str_std += '\t{:.5f}'.format(val)

            logger_test.info(
                f'\t{epoch} '
                f'\t{str_mean} \t{str_std} \t'
                f'\t{dice_nanmean_list.mean()} \t{dice_nanmean_list.std()} \t\t{dice_sum_list.mean()} \t{dice_sum_list.std()}'
                f'\t\t{avg_dc.mean()} \t{avg_dc.std()} \t{avg_hd.mean()} \t{avg_hd.std()} \t{avg_hd95.mean()}  \t{avg_hd95.std()}'
                f'\t{avg_prc.mean()} \t{avg_prc.std()} \t{avg_rcl.mean()} \t{avg_rcl.std()} \t{avg_sensi.mean()} \t{avg_sensi.std()}'
                f'\t{avg_speci.mean()} \t{avg_speci.std()} \t{avg_asd.mean()} \t{avg_asd.std()} '
                f'\t{avg_assd.mean()} \t{avg_assd.std()}\t{avg_jc.mean()} \t{avg_jc.std()}')

            logger_test_total.info(
                f'\t{epoch} \t{filel3} '
                f'\t{str_mean} \t{str_std} \t'
                f'\t{dice_nanmean_list.mean()} \t{dice_nanmean_list.std()} \t\t{dice_sum_list.mean()} \t{dice_sum_list.std()}'
                f'\t\t{avg_dc.mean()} \t{avg_dc.std()} \t{avg_hd.mean()} \t{avg_hd.std()} \t{avg_hd95.mean()}  \t{avg_hd95.std()}'
                f'\t{avg_prc.mean()} \t{avg_prc.std()} \t{avg_rcl.mean()} \t{avg_rcl.std()} \t{avg_sensi.mean()} \t{avg_sensi.std()}'
                f'\t{avg_speci.mean()} \t{avg_speci.std()} \t{avg_asd.mean()} \t{avg_asd.std()} '
                f'\t{avg_assd.mean()} \t{avg_assd.std()} \t{avg_jc.mean()} \t{avg_jc.std()}')

            logger_test.handlers.clear()
    logger_test_total.handlers.clear()


if __name__ == "__main__":

    # logger setup
    Log_name_config = "Log_config.log"
    Log_name_train = 'Log_train.log'
    Log_name_eval = 'Log_eval.log'

    Log_best_test = 'Log_best_test.log'

    # setup_logger(name, save_dir, filename="log.txt")
    logger_config = setup_logger('config', dir_datatype, filename=Log_name_config)
    logger_train = setup_logger('train', dir_datatype, filename=Log_name_train)
    logger_eval = setup_logger('eval', dir_datatype, filename=Log_name_eval)

    logger_config.info(cfg)
    cuda_check(logger_config)

    init_env(str(gpu_num))
    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU

    # for x in range(1, 11, 3):
    for x in [1, 4]:
        print("round(x*0.1, 1) : ", round(x * 0.1, 1))
        range_flow = round(x * 0.1, 1)
        dir_rangeflow = os.path.join(dir_datatype, str(range_flow))
        createFolder(dir_rangeflow)
        start_t = datetime.now()

        train_lvl1(dir_rangeflow, range_flow, load_model)
        train_lvl2(dir_rangeflow, range_flow, load_model)
        train_lvl3(dir_rangeflow, range_flow, load_model)
        # time
        end_t = datetime.now()
        total_t = end_t - start_t

        print("Time: ", total_t.total_seconds())
        logger_eval.info("Time: {}".format(total_t.total_seconds()))

        days = int(total_t.total_seconds()) // (60 * 60 * 24)
        days_left = int(total_t.total_seconds()) % (60 * 60 * 24)

        hours = days_left // (60 * 60)
        hours_left = days_left % (60 * 60)

        mins = hours_left // (60)
        mins_left = hours_left % (60)

        print("{}days {}hours {}mins {}secs".format(days, hours, mins, mins_left))
        logger_eval.info("{}days {}hours {}mins {}secs\n\n".format(days, hours, mins, mins_left))

        start_test = datetime.now()

        test(dir_rangeflow, range_flow, model_name_front=cfg.MODEL.NAME_FRONT, model_name_back=cfg.MODEL.NAME_BACK,
             doubleF=doubleF, doubleB=doubleB, max_only=False, brief=True, num_brief=10, num_save=10)
        end_test = datetime.now()
        total_test = end_test - start_test
        logger_eval.info("total_test time: {}".format(total_test))



