import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../..')
from ubc_AI.data import pfdreader
from itertools import chain
import cPickle

class Classify(object):
    def __init__(self, file):
        self.file = file
    def read_txt(self):
        with open(self.file, 'r') as f:
            pfd_found = np.genfromtxt(self.file, dtype=str)
            self.pfdfiles = pfd_found[..., 0]

            self.cands = np.asarray(pfd_found[..., 1], dtype=int)
            self.pulsars = np.where(self.cands == 1)
            self.harmonics = np.where(self.cands == 2)
            totalsize = len(self.cands)
            print '%s pfd found, %s pulsars, %s harmonics.' % (len(self.cands),
                                                               self.pulsars[0].size, self.harmonics[0].size)
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
            pfd_path = '/data/whf/AI/training/GBNCC_ARCC_rated/GBNCC_beams/'
            for pfd in self.pfdfiles:
                apfd = pfdreader(pfd_path + pfd)
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
                prob = (result[0][1]-0.26)/(0.74 - 0.26)
                self.predict.append(prob)
        def flatten(listOfLists):
            return chain.from_iterable(listOfLists)
        cand_scores = np.array(list(flatten([self.predict])))
        psr_scores = cand_scores[self.pulsars]
        harmonic_scores = cand_scores[self.harmonics]
        result = {'cand_scores': cand_scores, 'psr_scores': psr_scores, 'harmonic_scores': harmonic_scores,
                  'cands': self.cands, 'pulsars': self.pulsars, 'harmonics': self.harmonics, 'pfdfiles': self.pfdfiles}
        cPickle.dump(result, open('tmp/scores_ensemble_GBNCC.pkl', 'wb'), protocol=2)
        print 'done, result saved to ./scores_ensemble_GBNCC.pkl'

        fo = open('tmp/gbncc_prob.txt', 'w')
        for i, f in enumerate(self.pfdfiles):
            fo.write('%s %s %s\n' % (f, self.cands[i], self.predict[i]))
        fo.close()


if __name__ == "__main__":
    txt_file = '/data/whf/AI/training/ARCC_test/pfd_correct.txt'
    # txt_file = 'tmp/gbncc_100.txt'
    cls = Classify(txt_file)
    cls.read_txt()
    cls.classify_gbncc()






