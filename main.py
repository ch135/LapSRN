# -*- coding: utf-8 -*-
# @Time    : 2018/12/22 10:56
# @Author  : chenhao
# @FileName: main.py
# @Software: PyCharm
import argparse
import tensorlayer as tl
from config import *
import os, model, time
import numpy as np
import tensorflow as tf
from utils import *
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = config.train.batch_size
patch_size = config.train.in_patch_size
ni = int(np.sqrt(config.train.batch_size))


def compute_charboinner_loss(tensor1, tensor2, is_mean=True):
    epsilon = 1e-6
    if is_mean:
        loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(tensor1, tensor2) + epsilon)), [1, 2, 3]))
    else:
        loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(tensor1, tensor2) + epsilon)), [1, 2, 3]))
    return loss


def load_file_list():
    train_hr_list = []
    train_lr_list = []
    valid_hr_list = []
    valid_lr_list = []

    directory = config.train.hr_folder_path
    for filename in (y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory, y))):
        train_hr_list.append("%s%s" % (directory, filename))

    directory = config.train.lr_folder_path
    for filename in (y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory, y))):
        train_lr_list.append("%s%s" % (directory, filename))

    directory = config.valid.hr_folder_path
    for filename in (y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory, y))):
        valid_hr_list.append("%s%s" % (directory, filename))

    directory = config.valid.lr_folder_path
    for filename in (y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory, y))):
        valid_lr_list.append("%s%s" % (directory, filename))

    return sorted(train_hr_list), sorted(train_lr_list), sorted(valid_hr_list), sorted(valid_lr_list)


def prepare_nn_data(hr_image_list, lr_image_list, idx_umg=None):
    i = np.random.randint(len(hr_image_list)) if idx_umg is None else idx_umg

    input_image = get_iamge(lr_image_list[i])
    output_image = get_iamge(hr_image_list[i])
    scale = output_image.shape[0] // input_image.shape[0]
    assert scale == config.model.scale

    out_patch_size = patch_size * scale

    input_batch = np.empty([batch_size, patch_size, patch_size, 3])
    output_batch = np.empty([batch_size, out_patch_size, out_patch_size, 3])

    for i in range(batch_size):
        in_row_ind = np.random.randint(0, input_image.shape[0] - patch_size)
        in_col_ind = np.random.randint(0, input_image.shape[1] - patch_size)

        input_cropped = augment_imags(input_image[in_row_ind:in_row_ind + patch_size,
                                      in_col_ind:in_col_ind + patch_size])
        input_cropped = nomalize_iamge(input_cropped)
        input_cropped = np.expand_dims(input_cropped, axis=0)
        input_batch[i] = input_cropped

        out_row_ind = in_row_ind * scale
        out_col_ind = in_col_ind * scale

        out_cropped = augment_imags(output_image[out_row_ind:out_row_ind + out_patch_size,
                                    out_col_ind:out_col_ind + out_patch_size])
        out_cropped = nomalize_iamge(out_cropped)
        out_cropped = np.expand_dims(out_cropped, axis=0)
        output_batch[i] = out_cropped

    return input_batch, output_batch


def train():
    save_dir = "%s/%s" % (config.model.result_path, tl.global_flag["mode"])
    checkpoint_path = "%s" % (config.model.checkpoint_path)
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir(checkpoint_path)
    train_hr_list, train_lr_list, valid_hr_list, valid_lr_list = load_file_list()

    ##====================  DEFINE MODEL  =====================###
    t_image = tf.placeholder("float32", [batch_size, patch_size, patch_size, 3], name="t_image_input")
    t_target_image = tf.placeholder("float32",
                                    [batch_size, patch_size * config.model.scale, patch_size * config.model.scale, 3])
    t_target_image_dowm = tf.image.resize_images(t_target_image, size=[patch_size * 2, patch_size * 2], method=0,
                                                 align_corners=False)

    net_image2, net_grad2, net_image1, net_grad1 = model.LapSRN(t_image, is_train=True, reuse=False)
    loss1 = compute_charboinner_loss(net_image2.outputs, t_target_image, is_mean=True)
    loss2 = compute_charboinner_loss(net_image1.outputs, t_target_image_dowm, is_mean=True)
    g_loss = loss1 + loss2 * 4
    g_vars = tl.layers.get_variables_with_name("LapSRN", True, True)

    with tf.variable_scope("learning_rate"):
        lr_v = tf.Variable(config.train.lr_init, trainable=False)

    g_optim = tf.train.AdamOptimizer(lr_v, beta1=config.train.betal).minimize(g_loss, var_list=g_vars)

    ##=================  MODEL TEST  ========================###
    sample_ind = 37
    sample_input_imgs, sample_output_imgs = prepare_nn_data(valid_hr_list, valid_lr_list, sample_ind)
    tl.vis.save_images(truncate_images(sample_input_imgs), [ni, ni],
                       save_dir + '/train_sample_input.png')
    tl.vis.save_images(truncate_images(sample_output_imgs), [ni, ni],
                       save_dir + '/train_sample_output.png')

    net_image_test, net_grad_test, _, _ = model.LapSRN(t_image, is_train=False, reuse=True)

    ##================  RESTORE MMODEL =====================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_path + "/params_{}.npz".format(tl.global_flag["mode"]),
                                 network=net_image2)

    ##===============  TRAINING  =========================###
    sess.run(tf.assign(lr_v, config.train.lr_init))
    print("** learning rate %f" % config.train.lr_init)
    for epoch in range(config.train.n_epoch):
        # update learning rate
        if epoch != 0 and (epoch % config.train.decay_iter == 0):
            lr_decay = config.train.lr_decay ** (epoch // config.train.decay_iter)
            lr = lr_decay * config.train.lr_init
            sess.run(tf.assign(lr_v, lr))
            print("** learning rate %f " % lr)

        epoch_time = time.time()
        total_g_loss, n_iter = 0, 0

        # load image data
        for index in range(len(train_hr_list)):
            batch_input, batch_output = prepare_nn_data(train_hr_list, train_lr_list, index)
            errM, _ = sess.run([g_loss, g_optim], {t_image: batch_input, t_target_image: batch_output})
            total_g_loss += errM
            n_iter += 1
        print("[*] Epoch [%2d/%2d] loss: %.8f, time:%4.4fs" % (
            epoch, config.train.n_epoch, total_g_loss / n_iter, time.time() - epoch_time))

        # save model and evaluation on sample set
        if (epoch >= 0):
            tl.files.save_npz(net_image2.all_params,
                              name=checkpoint_path + "/params_{}.npz".format(tl.global_flag["mode"]), sess=sess)

            if config.train.dump_intermediate_result is True:
                valid_output, valid_grad_output = sess.run([net_image_test.outputs, net_grad_test.outputs],
                                                           {t_image: sample_input_imgs})
                tl.vis.save_images(truncate_images(valid_output), [ni, ni],
                                   save_dir + "/train_predict_%d.png" % epoch)
                tl.vis.save_images(truncate_images(np.abs(valid_grad_output)), [ni, ni],
                                   save_dir + '/train_predict_grad_%d.png' % epoch)

        if (epoch != 0 and epoch % 50 == 0):
            train_psnr(epoch)


def test(file, filename, dataset):
    try:
        img = get_iamge(file)
    except IOError:
        print("cannot open %s" % file)
    else:
        checkpoint_dir = config.model.checkpoint_path
        save_dir = "%s/%s%s%s" % (config.model.result_path, tl.global_flag["mode"], "/", dataset)
        psnr = None
        input_image = modcrop(img)
        input_image = nomalize_iamge(input_image)
        size = input_image.shape
        t_image = tf.placeholder("float32", [None, size[0], size[1], size[2]], name="input_image")
        net_image2, _, _, _ = model.LapSRN(t_image, is_train=False, reuse=tf.AUTO_REUSE)

        ##==================================== RESTORE G =======================================###
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            tl.layers.initialize_global_variables(sess)
            tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + "/params_train.npz", network=net_image2)

            # ==================================== TEST ============================================###
            out = sess.run(net_image2.outputs, {t_image: [input_image]})

            tl.files.exists_or_mkdir(save_dir)
            tl.vis.save_image(input_image, save_dir + "/test_input_%s" % filename)
            tl.vis.save_image(truncate_images(out[0, :, :, :]), save_dir + "/test_output_%s" % filename)
            image_bicubic(save_dir + "/test_output_%s" % filename, save_dir + "/test_output_%s" % filename)
            psnr = img_psnr(save_dir + "/test_input_%s" % filename, save_dir + "/test_output_%s" % filename)

        return psnr


if __name__ == "__main__":
    # python main.py -m test -f TESTIMAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["train", "test"], default="train", help="select mode")
    parser.add_argument("-p", "--path", help="input file path")

    args = parser.parse_args()
    tl.global_flag["mode"] = args.mode
    args.path = config.test.test_Data
    if tl.global_flag["mode"] == "train":
        train()
    elif tl.global_flag["mode"] == "test":
        write = pd.ExcelWriter(os.path.join(config.test.test_path, "result.xlsx"))
        df = None
        for dataset in os.listdir(args.path):
            all_psnr = 0.0
            start_time = time.time()
            result = []
            filenames = []
            data_path = "%s%s%s" % (args.path, dataset, "/")
            for filename in os.listdir(data_path):
                file = os.path.join(data_path, filename)
                psnr = test(file, filename, dataset)
                filenames.append(filename)
                result.append(psnr)
            mean = np.mean(result)
            alltime = time.time() - start_time
            result.append(mean)
            result.append(alltime)
            filenames.append("mean")
            filenames.append("time")

            df = pd.DataFrame([result], index=["PSNR"], columns=filenames)
            df.to_excel(write, dataset)
        write.save()
        write.close()
        # if (args.file is None):
        #     raise Exception("Please enter input file name for test mode")
        # else:
        #     test(args.file)
    else:
        raise Exception("Unknow --mode")
