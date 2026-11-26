
**Avviare training se si ha un file di config yaml:**
Sostituire a X il numero del training
python train.py --config configs/Training_tot.yaml



**Avviare analisi:**


for /L %S in (20000,20000,640000) do python evaluate.py --checkpoint runs/Training_tot/checkpoint_%S.pt --device cuda --num-envs 8 --num-workers 4 --episodes-per-context 100 --eval-seed 20261015 --mode greedy --output-dir evaluations/Training_tot/checkpoint_%S





**Avviare Plots:**


python -m analysis.analyze_training --evaluations-root evaluations/tot --training-metrics runs/Training_tot/metrics.jsonl --baseline evaluations/baseline_random --output-dir analysis/results/Training_tot --plots

