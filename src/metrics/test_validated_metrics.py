import unittest
import numpy as np
from validated_metrics import (compute_retrieval_metrics, compute_protein_centric_fmax, compute_term_aupr, protein_centric_f_at_threshold)

class TestMetrics(unittest.TestCase):
    def test_perfect(self):
        y=np.array([[1,0,1],[0,1,0]])
        s=np.array([[.9,.1,.8],[.2,.7,.1]])
        m=compute_retrieval_metrics(s,y)
        self.assertAlmostEqual(m['protein_wise_map'],1)
        self.assertAlmostEqual(m['pairwise_accuracy'],1)
        self.assertAlmostEqual(m['positive_block_coverage'],1)
        self.assertAlmostEqual(compute_protein_centric_fmax(y,s)[0],1)
        self.assertAlmostEqual(protein_centric_f_at_threshold(y,s,.5),1)
        self.assertAlmostEqual(compute_term_aupr(y,s),1)
    def test_ties(self):
        y=np.array([[1,0],[0,1]])
        s=np.ones((2,2))
        self.assertAlmostEqual(compute_retrieval_metrics(s,y)['pairwise_accuracy'],.5)
        self.assertAlmostEqual(compute_term_aupr(y,s),.5)
    def test_wrong_order(self):
        y=np.array([[1,0,0]])
        s=np.array([[.1,.9,.8]])
        self.assertAlmostEqual(compute_retrieval_metrics(s,y)['protein_wise_map'],1/3)
        self.assertAlmostEqual(compute_retrieval_metrics(s,y)['positive_block_coverage'],0)

if __name__=='__main__': unittest.main()
