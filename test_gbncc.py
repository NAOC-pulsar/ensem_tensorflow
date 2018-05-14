import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../..')
from ubc_AI.data import pfdreader

class Classify(object):
    def __init__(self, file):
        self.file = file
    def read_txt(self):
        with open(self.file, 'r') as f:
            data = np.genfromtxt(self.file, dtype=[('fn', '|S200'), ('label', int)])
            self.new_file = []
            self.label = np.vstack(data['label'])
            for file in data['fn']:
                self.new_file.append('/data/whf/AI/training/GBNCC_ARCC_rated/GBNCC_beams/' + file)
    def classify_gbncc(self):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt = tf.train.get_checkpoint_state('tmp/checkpoints/')
            new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            new_saver.restore(sess, ckpt.model_checkpoint_path)
            graph = tf.get_default_graph()
            fvp = graph.get_tensor_by_name('input1:0')
            tvp = graph.get_tensor_by_name('input2:0')
            dm = graph.get_tensor_by_name('input3:0')
            profile = graph.get_tensor_by_name('input4:0')
            rate = graph.get_tensor_by_name('rate:0')
            training = graph.get_tensor_by_name('is_training:0')
            y = graph.get_tensor_by_name('output:0')
            self.predict = []
            for pfd in self.new_file:
                apfd = pfdreader(pfd)
                res = apfd.getdata(intervals=64, subbands=64, phasebins=64, DMbins=64)
                data_TvP = np.empty([1, 64, 64, 1])
                data_TvP[0, :, :, 0] = res[0:4096].reshape((64, 64))
                data_Fvp = np.empty([1, 64, 64, 1])
                data_Fvp[0, :, :, 0] = res[4096:8192].reshape((64, 64))
                data_profile = np.empty([1, 64, 1])
                data_profile[0, :, 0] = res[8192:8256]
                data_dmb = np.empty([1, 64, 1])
                data_dmb[0, :, 0] = res[8256:8320]

                result = sess.run(y, feed_dict={fvp: data_Fvp, tvp: data_TvP, dm: data_dmb, profile: data_profile,
                                                rate: 0, training: False})
                self.predict.append(np.argmax(result, 1))

        pre_result = np.array(self.predict, dtype=int)
        count = np.equal(self.label, pre_result)
        right_res = np.sum(count)
        print right_res

if __name__ == "__main__":
    txt_file = 'tmp/gbncc_100.txt'
    cls = Classify(txt_file)
    cls.read_txt()
    cls.classify_gbncc()






