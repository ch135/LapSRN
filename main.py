# -*- coding: utf-8 -*-
# @Time    : 2018/12/22 10:56
# @Author  : chenhao
# @FileName: main.py
# @Software: PyCharm
import argparse
import tensorlayer as tl
import config
import os, utils, model, time
import numpy as np
import tensorflow as tf


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
    for filename in (y for y in os.list(directory) if os.path.isfile(os.path.join(directory, y))):
        train_hr_list.append("%s%s" % (directory, filename))

    directory = config.train.lr_folder_path
    for filename in (y for y in os.list(directory) if os.path.isfile(os.path.join(directory, y))):
        train_lr_list.append("%s%s" % (directory, filename))

    directory = config.valid.hr_floder_path
    for filename in (y for y in os.list(directory) if os.path.isfile(os.path.join(directory, y))):
        valid_hr_list.append("%s%s" % (directory, filename))

    directory = config.valid.lr_folder_path
    for filename in (y for y in os.list(directory) if os.path.isfile(os.path.join(directory, y))):
        valid_lr_list.append("%s%s" % (directory, filename))

    return sorted(train_hr_list), sorted(train_lr_list), sorted(valid_hr_list), sorted(valid_lr_list)


def prepare_nn_data(hr_image_list, lr_image_list, idx_umg=None):
    i = np.random.randint(len(hr_image_list)) if idx_umg is None else idx_umg

    input_image = utils.get_iamge(lr_image_list[i])
    output_image = utils.get_iamge(hr_image_list[i])
    scale = output_image.shape[0] / input_image.shape[0]
    assert scale == config.model.scale

    batch_size = config.train.batch_size
    patch_size = config.train.in_patch_size
    out_patch_size = patch_size * scale

    input_batch = np.empty([batch_size, patch_size, patch_size, 3])
    output_batch = np.empty([batch_size, out_patch_size, out_patch_size, 3])

    for i in range(batch_size):
        in_row_ind = np.random.randint(0, input_image.shape[0] - patch_size)
        in_col_ind = np.random.randint(0, input_image.shape[1] - patch_size)

        input_cropped = utils.augment_imags(input_image[in_row_ind:in_row_ind + patch_size,
                                            in_col_ind:in_col_ind + patch_size])
        input_cropped = utils.nomalize_iamge(input_cropped)
        input_cropped = np.expand_dims(input_cropped)
        input_batch[i] = input_cropped

        out_row_ind = in_row_ind * scale
        out_col_ind = in_col_ind * scale

        out_cropped = utils.augment_imags(output_image[out_row_ind:out_row_ind + out_patch_size,
                                          out_col_ind:out_col_ind + out_patch_size])
        out_cropped = utils.nomalize_iamge(out_cropped)
        out_cropped = np.expand_dims(out_cropped)
        output_batch[i] = out_cropped

    return input_batch, output_batch


def train():
    print("wait....")


def test(file):
    try:
        img = utils.get_iamge(file)
    except IOError:
        print("cannot open %s" % file)
    else:
        checkpoint_dir = config.model.checkpoint_path
        save_dir = "%s%s" % (config.model.result_path, tl.global_flag["mode"])

        input_image = utils.nomalize_iamge(img)
        size = input_image.shape
        t_image = tf.placeholder([None, size[0], size[1], size[2]], "float32", name="input_image")
        net_image2, _, _, _ = model.LapSRN(t_image, is_train=False, reuse=False)

        ##==================================== RESTORE G =======================================###
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + "/params_train.npz", network=net_image2)

        ##==================================== TEST ============================================###
        start_time = time.time()
        out = sess.run(net_image2, {t_image: [input_image]})
        print("took: %4.4fs" % (time.time() - start_time))

        tl.files.exists_or_mkdir(save_dir)
        tl.vis.save_image(utils.truncate_images(out[0, :, :, :]), save_dir="/test_out.png")
        tl.vis.save_image(input_image, save_dir="/test_input.png")


if __name__ == "__main__":
    # python main.py -m test -f TESTIMAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "-mode", choices=["train", "test"], default="train", help="select mode")
    parser.add_argument("-f", "--file", help="input file")

    args = parser.parse_args()

    tl.global_flag["mode"] = args.mode
    if tl.global_flag["mode"] == "train":
        train()
    elif tl.global_flag["mode"] == "test":
        test()
    else:
        raise Exception("Unknow --mode")
