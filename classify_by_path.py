import glob
import tensorflow as tf
import numpy as np
import sys
import time
sys.path.append("..")
sys.path.append("../..")
from ubc_AI.data import pfdreader
from ProgressBar import progressBar as PB


class Classify(object):
    def __init__(self, path, txt_file):
        self.path = path
        self.txt_file = txt_file

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
            # str_info = []
            t_1 = time.time()

            with open(self.txt_file, 'r') as f:
                processed_pfds = np.genfromtxt(self.txt_file, dtype=[('fn', '|S200'), ('s', 'f')])
                processed_pfds_set = set(processed_pfds['fn'])
                print 'we already classified ', len(processed_pfds_set), 'files'

            with open(self.txt_file, 'a') as f:

                allpfds = glob.glob(self.path + '*.pfd')
                pb = PB(maxValue=len(allpfds))
                allpfds_set = set(allpfds)
                to_process_set = allpfds_set - processed_pfds_set
                print len(to_process_set), 'files to be classified.'

                for i, pfd in enumerate(sorted(list(to_process_set))):
                    #print pfd 
                    apfd = pfdreader(pfd)

                    #TvP = apfd.getdata(intervals=64).reshape(64, 64)
                    #new_TvP = np.array(TvP)
                    #data_TvP = np.empty([1, 64, 64, 1])
                    #data_TvP[0, :, :, 0] = new_TvP
                    #FvP = apfd.getdata(subbands=64).reshape(64, 64)
                    #new_FvP = np.array(FvP)
                    #data_FvP = np.empty([1, 64, 64, 1])
                    #data_FvP[0, :, :, 0] = new_FvP
                    #profile = apfd.getdata(phasebins=64)
                    #new_profile = np.array(profile)
                    #data_profile = np.empty([1, 64, 1])
                    #data_profile[0, :, 0] = np.transpose(new_profile)
                    #dmb = apfd.getdata(DMbins=64)
                    #new_dmb = np.array(dmb)
                    #data_dmb = np.empty([1, 64, 1])
                    #data_dmb[0, :, 0] = np.transpose(new_dmb)

                    res = apfd.getdata(intervals=64, subbands=64, phasebins=64, DMbins=64)

                    data_TvP = np.empty([1, 64, 64, 1])
                    data_TvP[0, :, :, 0] = res[0:4096].reshape((64,64))
                    data_FvP = np.empty([1, 64, 64, 1])
                    data_FvP[0, :, :, 0] = res[4096:8192].reshape((64,64))
                    data_profile = np.empty([1, 64, 1])
                    data_profile[0, :, 0] = res[8192:8256]
                    data_dmb = np.empty([1, 64, 1])
                    data_dmb[0, :, 0] = res[8256:8320]

                    result = sess.run(y, feed_dict={fvp: data_FvP, tvp: data_TvP, dm: data_dmb, pro: data_profile,
                                                    rate: 0, training: False})
                    # label = np.argmax(result, 1)
                    proba = np.float32(result[0][1])
                    str_info = pfd + ' ' + str(proba) + '\n'
                    # str_info = pfd + ': ' + str(label) + '\n'
                    f.write(str_info)

                    if i % 10 == 0:
                        pb(i)

            t_2 = time.time() - t_1
            print('Classifying complete in {:.0f} m {:.0f} s'.format(t_2 // 60, t_2 % 60))

if __name__ == '__main__':

    path = '/data/public/AI_data/PRESTO/*/'
    txt_file = 'tmp/fast_zhu_result.txt'
    # path = '/data/whf/AI/training/GBNCC_ARCC_rated/'
    cls = Classify(path, txt_file)
    cls.classify_by_path()
