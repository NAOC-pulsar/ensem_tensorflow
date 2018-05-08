import glob
import tensorflow as tf
import numpy as np
import sys
import time
sys.path.append("..")
sys.path.append("../..")
from ubc_AI.data import pfdreader


class Classify(object):
    def __init__(self, path):
        self.path = path

    def classify_by_path(self):

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('tmp/checkpoints/')
            new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            new_saver.restore(sess, ckpt.model_checkpoint_path)
            graph = tf.get_default_graph()
            fvp = graph.get_tensor_by_name('input1:0')
            tvp = graph.get_tensor_by_name('input2:0')
            dm = graph.get_tensor_by_name('input3:0')
            pro = graph.get_tensor_by_name('input4:0')
            rate = graph.get_tensor_by_name('rate:0')
            training = graph.get_tensor_by_name('is_training:0')
            y = graph.get_tensor_by_name('output:0')
            str_info = []
            t_1 = time.time()
            for file in glob.glob(self.path + '*.pfd'):
                print file
                apfd = pfdreader(file)
                TvP = apfd.getdata(intervals=64).reshape(64, 64)
                new_TvP = np.array(TvP)
                data_TvP = np.empty([1, 64, 64, 1])
                data_TvP[0, :, :, 0] = new_TvP
                FvP = apfd.getdata(subbands=64).reshape(64, 64)
                new_FvP = np.array(FvP)
                data_FvP = np.empty([1, 64, 64, 1])
                data_FvP[0, :, :, 0] = new_FvP
                profile = apfd.getdata(phasebins=64)
                new_profile = np.array(profile)
                data_profile = np.empty([1, 64, 1])
                data_profile[0, :, 0] = np.transpose(new_profile)
                dmb = apfd.getdata(DMbins=64)
                new_dmb = np.array(dmb)
                data_dmb = np.empty([1, 64, 1])
                data_dmb[0, :, 0] = np.transpose(new_dmb)
                result = sess.run(y, feed_dict={fvp: data_FvP, tvp: data_TvP, dm: data_dmb, pro: data_profile,
                                                rate: 0, training: False})
                # label = np.argmax(result, 1)
                proba = np.float32(result[0][1])
                str_info = file + '.png ' + str(proba) + '\n'
                # str_info = file + ': ' + str(label) + '\n'
                with open('tmp/fast_zhu_result.txt', 'a') as f:
                    f.write(str_info)
            t_2 = time.time() - t_1
            print('Classifying complete in {:.0f} m {:.0f} s'.format(t_2 // 60, t_2 % 60))

if __name__ == '__main__':

    path = '/data/public/AI_data/PRESTO/*/'
    # path = '/data/whf/AI/training/GBNCC_ARCC_rated/'
    cls = Classify(path)
    cls.classify_by_path()