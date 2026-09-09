"""Safety and mathematical checks; no original data or models are mutated."""
import unittest
import numpy as np
import pandas as pd
import torch

from run_experiment import (REPO, TREATMENTS, capped_copy, fit_cap, training_target,
                            inputs, inside_repo, eligible, metrics, block_ci)
from london_unet_model import LondonResidualUNet, parameter_count


class CappingTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.array([[[5., 50., np.nan, 70.]], [[10., 90., 40., 999.]], [[10000., 10000., 10000., 10000.]]], np.float32)
        self.w = np.ones_like(self.raw)
        self.w[1,0,3] = 0
        self.prep = {"channels": [0,1], "feature_centres": [0.,0.], "feature_scales": [1.,1.],
                     "target_log_mean": 2., "target_log_std": .5, "lds_edges": [0.,25.,100.], "lds_lookup": [1.,2.5]}

    def test_cap_uses_training_only_and_excludes_zero_weight(self):
        values = np.array([5.,50.,70.,10.,90.,40.])
        result = fit_cap(self.raw, self.w, 2, .995)
        self.assertAlmostEqual(result["cap_pm25"], np.quantile(values,.995))
        self.assertEqual(result["training_cell_days"],6)

    def test_future_targets_do_not_change_cap(self):
        first = fit_cap(self.raw, self.w, 2, .99)
        other = self.raw.copy(); other[2] = -10000
        self.assertEqual(first,fit_cap(other,self.w,2,.99))

    def test_cap_only_upper_observed_values_and_preserves_original(self):
        original = self.raw.copy()
        observed = np.isfinite(self.raw) & (self.w>0)
        result = capped_copy(self.raw,observed,43.)
        np.testing.assert_equal(self.raw,original)
        self.assertTrue(np.isnan(result[0,0,2]))
        self.assertEqual(result[0,0,0],5.)
        self.assertEqual(result[0,0,1],43.)
        self.assertEqual(result[1,0,3],999.)
        self.assertFalse(np.shares_memory(result,self.raw))

    def test_uncapped_control_identical(self):
        np.testing.assert_equal(capped_copy(self.raw,np.isfinite(self.raw),None),self.raw)
        self.assertEqual(fit_cap(self.raw,self.w,2,None)["affected_cell_days"],0)

    def test_weights_and_masks_do_not_depend_on_cap(self):
        laqn = np.array([[1.,np.nan,np.nan,1.]],np.float32)
        y0,w0=training_target(self.raw[0],self.w[0],laqn,self.prep,None)
        y1,w1=training_target(self.raw[0],self.w[0],laqn,self.prep,43.)
        np.testing.assert_array_equal(w0,w1)
        self.assertEqual(y0[0,2],0.); self.assertEqual(y1[0,2],0.)
        self.assertEqual(w0[0,2],0.)
        self.assertEqual(y0[0,0],y1[0,0])
        self.assertLess(y1[0,1],y0[0,1])

    def test_hybrid_residual_and_standalone_direct_target(self):
        laqn=np.ones((1,4)); baseline=np.full((1,4),.4,np.float32)
        direct,w=training_target(self.raw[0],self.w[0],laqn,self.prep,43.)
        residual,w2=training_target(self.raw[0],self.w[0],laqn,self.prep,43.,baseline)
        valid=w>0
        np.testing.assert_allclose(residual[valid],direct[valid]-.4,atol=1e-6)
        np.testing.assert_array_equal(w,w2)
        self.assertEqual(residual[0,2],0.)

    def test_inputs_read_only_and_no_hgb_channel_for_standalone(self):
        x=np.ones((2,2,3,4),np.float16); original=x.copy()
        base=np.full((2,3,4),7.,np.float32)
        plain=inputs(x,[0],self.prep,None)
        hybrid=inputs(x,[0],self.prep,base)
        self.assertEqual(plain.shape,(1,2,3,4))
        self.assertEqual(hybrid.shape,(1,3,3,4))
        np.testing.assert_array_equal(hybrid[:, -1],7.)
        np.testing.assert_array_equal(x,original)

    def test_unknown_quantile_rejected(self):
        with self.assertRaises(ValueError): fit_cap(self.raw,self.w,2,.5)

    def test_all_masked_rejected(self):
        with self.assertRaises(ValueError): fit_cap(self.raw,np.zeros_like(self.w),2,.99)

    def test_output_paths(self):
        self.assertEqual(inside_repo(REPO/'artifacts'/'test'),(REPO/'artifacts'/'test').resolve())
        for path in [REPO,REPO.parent/'outside',REPO/'artifacts'/'..'/'..'/'outside',REPO/'.git'/'test']:
            with self.assertRaises(ValueError): inside_repo(path)

    def test_validation_metrics_do_not_cap_observations(self):
        m=metrics([10.,40.],[10.,20.])
        self.assertEqual(m['high']['mae'],20.)
        self.assertEqual(m['mae'],10.)
        self.assertAlmostEqual(m['rmse'],np.sqrt(200))
        self.assertEqual(m['high_recall'],0.)

    def test_gate_rejects_tradeoff_and_accepts_all_improved(self):
        control={'rmse':4.,'mae':2.4,'high':{'mae':14.}}
        good={'rmse':3.9,'mae':2.3,'high':{'mae':13.9}}
        bad={'rmse':3.9,'mae':2.3,'high':{'mae':14.1}}
        self.assertTrue(eligible(good,control))
        self.assertFalse(eligible(bad,control))
        self.assertFalse(eligible(control,control))

    def test_paired_block_interval_identity_and_direction(self):
        y=np.linspace(3.,35.,42)
        f=pd.DataFrame({'date_idx':np.repeat(np.arange(21),2),'observed_pm25':y})
        same=block_ci(f,y+1,y+1,100)
        self.assertEqual(same['ci95'],[0.,0.])
        better=block_ci(f,y+2,y+1,100)
        np.testing.assert_allclose(better['ci95'],[-1.,-1.])

    def test_nonfinite_evaluation_rejected(self):
        with self.assertRaises(ValueError): metrics([10.,40.],[10.,np.nan])

    def test_backbone_unchanged_in_channels(self):
        plain=LondonResidualUNet(66,32,.12)
        hybrid=LondonResidualUNet(67,32,.12)
        self.assertEqual(plain.encoder1.conv1.in_channels,66)
        self.assertEqual(hybrid.encoder1.conv1.in_channels,67)
        self.assertEqual(parameter_count(hybrid),1932801)
        self.assertTrue(torch.equal(plain.head.weight,torch.zeros_like(plain.head.weight)))


if __name__ == '__main__':
    unittest.main()
