"""Read-only model instrumentation for the 10-qubit diagnostic study.

No model source is changed. Exact features call the production simulator.
Full-history circuits are analysis artifacts, not backend submission code.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import importlib.metadata

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from quantum_reservoir_qiskit import (
    generate_coupling_matrix, load_coupling_matrices, build_ising_hamiltonian,
    compute_unitaries, quantum_reservoir, encode_input, build_z_observables, MIN_RV, DIF,
)
from quantum_reservoir_trotter import build_trotter_step, build_trotter_evolution
from qrcstudy.extended_models import QR1, QR2
from qrcstudy.report import load_validated

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/snapshots/extended-2026-09-24-v2/monthly.csv'
RUN = ROOT / 'results/extended-2026-09-24'
MODELS = {'QR1': QR1, 'QR2': QR2}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def validated_predictions():
    # The existing validator rewrites predictions.csv: run it on a temporary copy.
    before = digest(RUN / 'predictions.csv')
    with tempfile.TemporaryDirectory(prefix='qrc-diagnostic-validation-') as folder:
        for name in ['manifest.json', 'report_receipt.json', 'predictions.csv']:
            shutil.copy2(RUN / name, Path(folder) / name)
        frame, config = load_validated(folder)
    assert digest(RUN / 'predictions.csv') == before
    return frame, config


def history_circuit(values, J, steps, virtual_nodes, endpoint):
    """One independently repeated measurement endpoint, with two input resets."""
    qc = QuantumCircuit(10, 10)
    u, du = build_trotter_evolution(10, J, 1., steps, virtual_nodes)
    for lag in range(3):
        if lag:
            qc.reset(range(7))
        for q, value in enumerate(values[lag]):
            qc.ry(float(np.pi * value), q)
        if lag < 2:
            qc.compose(u, inplace=True)
        else:
            for _ in range(endpoint):
                qc.compose(du, inplace=True)
    qc.measure(range(10), range(10))
    return qc


def instrument(values, u, du, virtual_nodes):
    """Mirror the production three-input protocol and retain diagnostic states."""
    hidden = DensityMatrix.from_label('000')
    obs = build_z_observables(10)
    output, probabilities, states = [], [], []
    for lag in range(3):
        joint = hidden.tensor(encode_input(values[lag], 7))
        if lag < 2:
            joint = joint.evolve(u)
            hidden = partial_trace(joint, list(range(7)))
            reduced = hidden.data
            states.append((lag + 1, float(np.trace(reduced @ reduced).real),
                           float(np.linalg.eigvalsh(reduced).min()),
                           float(abs(np.trace(reduced) - 1)),
                           float(np.max(abs(reduced - reduced.conj().T)))))
        else:
            for node in range(virtual_nodes):
                joint = joint.evolve(du)
                output.extend(float(joint.expectation_value(z).real) for z in obs)
                probabilities.append(joint.probabilities())
            reduced = partial_trace(joint, list(range(7))).data
            states.append((3, float(np.trace(reduced @ reduced).real),
                           float(np.linalg.eigvalsh(reduced).min()),
                           float(abs(np.trace(reduced) - 1)),
                           float(np.max(abs(reduced - reduced.conj().T)))))
    return np.array(output), np.array(probabilities), states


def compute(out):
    out = Path(out)
    tables = out / 'tables'; tables.mkdir(parents=True, exist_ok=True)
    cache = out / 'cache'; cache.mkdir(exist_ok=True)
    started = time.time()
    frame = pd.read_csv(DATA, index_col=0, parse_dates=True)
    assert frame.index.is_unique and frame.index.equals(pd.date_range(frame.index[0], frame.index[-1], freq='ME'))
    assert np.allclose(frame.log_rv, (frame.RV + 1) * DIF + MIN_RV, atol=1e-12, rtol=0)
    pred, config = validated_predictions()
    pred.to_csv(tables / 'validated_predictions.csv', index=False)
    old = pd.read_csv(ROOT / 'data/Data.CSV', index_col=0)
    original = pd.read_csv(ROOT / '1950-2026.csv', index_col=0)
    overlap_error = float(np.max(np.abs(old.to_numpy() - original.iloc[:len(old)].to_numpy())))
    assert overlap_error < 1e-12
    legacy = pd.read_csv(ROOT / 'data/predict_result.csv')
    reference = pd.read_csv(ROOT / 'results/predictions/qrc_predict_result.csv')
    historical_error = {m: float(np.max(abs(legacy[m] - reference[m]))) for m in MODELS}
    provenance = {'repository': 'aidanccc/qrc-volatility-research',
        'branch': subprocess.check_output(['git','branch','--show-current'], cwd=ROOT, text=True).strip(),
        'baseline_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
        'input_hashes': {str(p.relative_to(ROOT)): digest(p) for p in [DATA, RUN/'predictions.csv', RUN/'manifest.json', ROOT/'data/coeff_10.jld2', ROOT/'data/Data.CSV',ROOT/'1950-2026.csv',ROOT/'data/predict_result.csv',ROOT/'results/predictions/qrc_predict_result.csv']},
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in list((ROOT/'analysis/diagnostics').glob('*.py')) + [ROOT/'quantum_reservoir_qiskit.py', ROOT/'quantum_reservoir_trotter.py', ROOT/'qrcstudy/models.py', ROOT/'qrcstudy/extended_models.py']},
        'versions': {p: importlib.metadata.version(p) for p in ['qiskit','numpy','scipy','pandas','matplotlib','h5py']},
        'seed': 0, 'tau': 1., 'qubits': 10, 'input_qubits': 7, 'hidden_qubits': 3, 'history_months': 3,
        'trotter_steps': [1,2,4,8], 'shot_counts': [100,1000,10000],
        'forecast_scope': {'seeds': config['seeds'], 'windows': config['windows']},
        'couplings': 'Extended: generate_coupling_matrix(10, seed=0), shared QR1/QR2. Historical: original JLD2 matrices 0 and 1.',
        'historical_data_max_abs_difference': overlap_error,
        'historical_published_predictions_max_abs_difference': historical_error,
        'historical_check_scope': 'Existing Python versus author prediction artifacts; no new full historical forecast rerun.',
        'feature_scope': 'Seed 0, full valid history; published forecast diagnostics use seeds 0-4.',
        'backend_execution': 'None: ideal density matrix, logical circuit inspection, analytic shot uncertainty only.'}
    write_json(out/'manifest.json', provenance)
    # The selected target dates are determined by input availability, never target outcomes.
    candidates = []
    union = list(dict.fromkeys(QR1 + QR2))
    for t in np.flatnonzero(frame.index >= '2018-01-01'):
        if np.isfinite(frame[union].iloc[t-3:t].to_numpy()).all(): candidates.append(int(t))
    selected = np.array(candidates)[np.linspace(0,len(candidates)-1,12).round().astype(int)]
    pd.DataFrame([{'target_month': str(frame.index[t].date()), 'input_start': str(frame.index[t-3].date()),
                   'input_end': str(frame.index[t-1].date()),'position': t} for t in selected]).to_csv(tables/'selected_histories.csv', index=False)
    j = generate_coupling_matrix(10, seed=0)
    pd.DataFrame(j).to_csv(tables/'extended_couplings_seed0.csv',index=False)
    matrices = load_coupling_matrices(str(ROOT/'data/coeff_10.jld2'))
    assert matrices is not None and matrices.shape == (100,10,10)
    for i,m in enumerate(MODELS):pd.DataFrame(matrices[i]).to_csv(tables/f'historical_couplings_{m}.csv',index=False)
    h = build_ising_hamiltonian(10,j)
    # Executable check of Qiskit's chronological gate order vs the docstring formula.
    small_j = generate_coupling_matrix(2, seed=0);dt=.3
    hx = build_ising_hamiltonian(2,small_j)-build_ising_hamiltonian(2,np.zeros((2,2)))
    hz = build_ising_hamiltonian(2,np.zeros((2,2)))
    small_u=Operator(build_trotter_step(2,small_j,dt)).data
    order_error = float(np.max(abs(small_u-expm(-1j*dt*hz)@expm(-1j*dt*hx))))
    reversed_error = float(np.max(abs(small_u-expm(-1j*dt*hx)@expm(-1j*dt*hz))))
    assert order_error < 1e-12
    # Qubit-order and encoding checks with a known computational-basis state.
    probe = DensityMatrix.from_label('000').tensor(encode_input(np.array([1.,0,0,0,0,0,0]),7))
    probe_z = np.array([probe.expectation_value(z).real for z in build_z_observables(10)])
    assert np.allclose(probe_z, [-1]+[1]*9)
    assert np.allclose(partial_trace(probe,list(range(7))).data, DensityMatrix.from_label('000').data)
    records, errors, resources, shot_rows, condition, feature_summary, matches, legacy_checks = [],[],[],[],[],[],[],[]
    bits = ((np.arange(1024)[:,None] >> np.arange(10)) & 1)
    signs = 1 - 2*bits
    for model,cols in MODELS.items():
        v = 1 if model == 'QR1' else 2
        u,du = compute_unitaries(h,1.,v)
        x=frame[cols].to_numpy();bad=np.flatnonzero(~np.isfinite(x).all(axis=1));n=int(bad[0]) if len(bad) else len(x)
        feature_file=cache/f'{model}-seed0.npy'; receipt=feature_file.with_suffix('.json')
        identity={'data_sha256': digest(DATA), 'source_sha256':digest(ROOT/'quantum_reservoir_qiskit.py'), 'columns':cols,'seed':0,'valid_prefix':n}
        if feature_file.exists():
            metadata=json.loads(receipt.read_text());assert metadata['identity']==identity and metadata['sha256']==digest(feature_file)
            features=np.load(feature_file)
        else:
            print(f'{model}: regenerating {n-2} exact forecast feature vectors',flush=True)
            features=quantum_reservoir(pd.DataFrame(np.vstack([x[:n], np.zeros((1,7))]),columns=cols), cols,u,du,3,v,10).T
            np.save(feature_file,features);write_json(receipt,{'identity':identity,'sha256':digest(feature_file)})
        assert features.shape==(n+1,10*v) and np.isfinite(features).all()
        assert np.max(abs(features[3:]))<=1+1e-10
        dates=frame.index.append(pd.DatetimeIndex([frame.index[-1]+pd.offsets.MonthEnd()]))[:n+1]
        names=[f'v{node+1}_q{q}' for node in range(v) for q in range(10)]
        pd.DataFrame(features[3:],index=dates[3:],columns=names).rename_axis('target_month').to_csv(tables/f'{model}_features_seed0.csv')
        for name,var in zip(names,np.var(features[3:],axis=0)):
            feature_summary.append({'model':model,'feature':name,'variance':var})
        for w in [120,571]:
            subset=pred.loc[(pred.model==model)&(pred.seed==0)&(pred.window==w)&pred.status.eq('ok')]
            for row in subset.itertuples():
                if row.target_month not in dates:continue
                t=dates.get_loc(row.target_month)
                a=features[t-w:t];y=frame.RV.iloc[t-w:t].to_numpy()
                weights=np.linalg.solve(a.T@a+1e-8*np.eye(10*v),a.T@y)
                estimate=(float(features[t]@weights)+1)*DIF+MIN_RV
                matches.append({'model':model,'window':w,'target_month':str(row.target_month.date()),'absolute_difference':abs(estimate-row.predicted_log_rv)})
                condition.append({'model':model,'window':w,'target_month':str(row.target_month.date()),'condition':float(np.linalg.cond(a)),'weight_norm':float(np.linalg.norm(weights))})
        exact={}
        for t in selected:
            values=x[t-3:t];z,prob,states=instrument(values,Operator(u),Operator(du),v)
            assert np.allclose(z,features[t],atol=1e-11,rtol=0)
            assert np.min(prob)>-1e-12 and np.allclose(prob.sum(axis=1),1,atol=1e-10)
            exact[t]=(z,prob)
            for stage,purity,mineig,traceerr,hermerr in states:
                assert mineig>-1e-10 and traceerr<1e-10 and hermerr<1e-10 and 1/8-1e-10<=purity<=1+1e-10
                records.append({'model':model,'target_month':str(frame.index[t].date()),'stage':stage,'hidden_purity':purity,'min_eigenvalue':mineig,'trace_error':traceerr,'hermiticity_error':hermerr})
            for shots in [100,1000,10000]:
                for node in range(v):
                    means=z[node*10:(node+1)*10]
                    covariance=(signs.T@(prob[node,:,None]*signs)-np.outer(means,means))/shots
                    for q in range(10):
                        shot_rows.append({'model':model,'target_month':str(frame.index[t].date()),'virtual_node':node+1,'qubit':q,'shots':shots,'z_mean':means[q],'standard_error':float(np.sqrt(max(0,covariance[q,q])))})
        # Original matrix provenance check on one historical three-month input window.
        hu,hdu=compute_unitaries(build_ising_hamiltonian(10,matrices[v-1]),1.,v)
        historic_values=old[cols].iloc[570:573].to_numpy()
        z,_,_=instrument(historic_values,Operator(hu),Operator(hdu),v)
        direct=quantum_reservoir(pd.DataFrame(np.vstack([historic_values,np.zeros((1,7))]),columns=cols),cols,hu,hdu,3,v,10)[:,3]
        legacy_checks.append({'model':model,'max_abs_instrumentation_error':float(np.max(abs(z-direct)))})
        for steps in [1,2,4,8]:
            print(f'{model}: measuring Trotter steps={steps} on 12 histories',flush=True)
            uc,dc=build_trotter_evolution(10,j,1.,steps,v)
            uop,dop=Operator(uc),Operator(dc)
            for t in selected:
                z,_,_=instrument(x[t-3:t],uop,dop,v)
                for idx,(a,b) in enumerate(zip(exact[t][0],z)):
                    errors.append({'model':model,'target_month':str(frame.index[t].date()),'steps':steps,'virtual_node':idx//10+1,'qubit':idx%10,'exact_z':a,'trotter_z':b,'absolute_error':abs(a-b)})
            for endpoint in range(1,v+1):
                qc=history_circuit(x[selected[0]-3:selected[0]],j,steps,v,endpoint);ops=qc.count_ops()
                resources.append({'model':model,'steps':steps,'endpoint':endpoint,'qubits':qc.num_qubits,'depth':qc.depth(),'ry':ops.get('ry',0),'rz':ops.get('rz',0),'rxx':ops.get('rxx',0),'reset':ops.get('reset',0),'measure':ops.get('measure',0),'evolution_block_rxx':uc.count_ops().get('rxx',0),'evolution_block_depth':uc.depth()})
        if model=='QR1':
            example=pd.DataFrame(x[selected[0]-3:selected[0]],index=frame.index[selected[0]-3:selected[0]],columns=cols)
            example.rename_axis('input_month').to_csv(tables/'example_three_month_inputs.csv')
            (example*np.pi).rename_axis('input_month').to_csv(tables/'example_three_month_angles.csv')
    for name,rows in [('hidden_states',records),('trotter_observable_errors',errors),('circuit_resources',resources),('shot_uncertainty',shot_rows),('rolling_conditioning',condition),('feature_variance',feature_summary),('forecast_reproduction',matches),('historical_instrumentation',legacy_checks)]:
        pd.DataFrame(rows).to_csv(tables/f'{name}.csv',index=False)
    max_match=max(r['absolute_difference'] for r in matches)
    assert max_match<1e-9, max_match
    validation={'published_forecasts_validated':True,'reproduced_seed0_forecasts':len(matches),'max_forecast_difference':max_match,'historical_instrumentation_max_difference':max(r['max_abs_instrumentation_error'] for r in legacy_checks),'gate_order_actual_Z_after_XX_error':order_error,'gate_order_docstring_reversed_error':reversed_error,'qubit_order_probe':probe_z.tolist(),'hidden_states_valid':True,'sampled_histories_per_model':12,'elapsed_seconds':time.time()-started}
    write_json(out/'validation.json',validation)
    provenance['elapsed_seconds']=time.time()-started;write_json(out/'manifest.json',provenance)
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=1):compute(args.output)
