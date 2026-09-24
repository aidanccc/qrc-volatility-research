import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from qrcstudy.colin_data import derived,read_monthly,affine_map,prepare
from qrcstudy.colin_models import inputs,predict,QR1,QR2
from qrcstudy.models import sequences
from quantum_reservoir_qiskit import MIN_RV,DIF
ROOT=Path(__file__).resolve().parents[1]

class ColinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=read_monthly(ROOT/'1950-2026.csv')
        cls.prepared=read_monthly(ROOT/'data/snapshots/colin-2026-09-24-v2/monthly.csv')

    def test_supplied_missing_block_and_prepared_gaps(self):
        self.assertEqual(int(self.raw.loc['2018':,['MKT','SMB','HML','STR']].isna().sum().sum()),416)
        self.assertEqual(int(self.prepared.isna().sum().sum()),4)
        self.assertTrue(self.prepared.loc['2026-07-31'].notna().all())

    def test_history_and_derived_identities(self):
        old=read_monthly(ROOT/'data/Data.CSV')
        np.testing.assert_allclose(self.raw.loc[:'2017'],old,atol=2e-16,rtol=0)
        out,changes,_=derived(self.raw)
        self.assertEqual(len(changes),211)
        np.testing.assert_allclose(out.RV1.iloc[1:],out.RV.shift().iloc[1:],atol=1e-10)
        np.testing.assert_allclose(out.RV2.iloc[2:],out.RV.shift(2).iloc[2:],atol=1e-10)

    def test_future_cannot_change_past(self):
        f=self.raw.copy();f.loc['2020':,'RV']+=2
        a,_,_=derived(self.raw);b,_,_=derived(f)
        pd.testing.assert_frame_equal(a.loc[:'2019'],b.loc[:'2019'])
        for model in ['HAR','HARX','QR1','QR2','LSTMX']:
            pd.testing.assert_frame_equal(inputs(a,model).loc[:'2019'],inputs(b,model).loc[:'2019'])

    def test_feature_sets_and_inverse(self):
        self.assertEqual(inputs(self.prepared,'QR1').columns.tolist(),QR1)
        self.assertEqual(inputs(self.prepared,'QR2').columns.tolist(),QR2)
        self.assertEqual(inputs(self.prepared,'LSTMX').shape[1],11)
        self.assertEqual(inputs(self.prepared,'ARMAX').shape[1],10)
        np.testing.assert_allclose((self.prepared.log_rv-MIN_RV)/DIF-1,self.prepared.RV,atol=1e-12)

    def test_missing_origin_fails_and_observed_month_works(self):
        f=self.prepared;x=inputs(f,'HARX').to_numpy();y=np.append(f.RV,np.nan)
        self.assertTrue(np.isfinite(predict('HARX',x,y,None,None,len(f)-1,120,0,1)))
        with self.assertRaisesRegex(ValueError,'Unavailable'):
            predict('HARX',x,y,None,None,len(f),120,0,1)
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            predict('HAR',x,y,None,None,20,120,0,1)

    def test_current_and_future_targets_cannot_change_forecast(self):
        f=self.prepared;x=inputs(f,'HAR').to_numpy();y=f.RV.to_numpy().copy();t=850
        a=predict('HAR',x,y,None,None,t,120,0,1)
        g=f.copy();g.iloc[t:,g.columns.get_loc('RV')]+=5
        b=predict('HAR',inputs(g,'HAR').to_numpy(),g.RV.to_numpy(),None,None,t,120,0,1)
        self.assertAlmostEqual(a,b,places=12)

    def test_mapping_rejects_wrong_definition_and_ignores_future(self):
        idx=pd.date_range('1950-01-31','2026-08-31',freq='ME');rng=np.random.default_rng(1)
        raw=pd.Series(np.round(rng.normal(size=len(idx)),2),index=idx);norm=.05*raw-.1
        first=affine_map(raw,norm);self.assertTrue(first['accepted'])
        norm.loc['2018':]=100
        self.assertEqual(first,affine_map(raw,norm))
        wrong=raw**2
        self.assertFalse(affine_map(raw,wrong)['accepted'])

    def test_incomplete_month_rejected_before_download(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'incomplete month'):
                prepare(ROOT/'1950-2026.csv',Path(d)/'snapshot',as_of='2026-08-24')

    def test_invalid_calendar_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'f.csv';self.raw.iloc[[0,2]].to_csv(p)
            with self.assertRaisesRegex(ValueError,'gap'):read_monthly(p)
            pd.concat([self.raw.iloc[:1],self.raw.iloc[:1]]).to_csv(p)
            with self.assertRaisesRegex(ValueError,'Duplicate'):read_monthly(p)

if __name__=='__main__':unittest.main()
