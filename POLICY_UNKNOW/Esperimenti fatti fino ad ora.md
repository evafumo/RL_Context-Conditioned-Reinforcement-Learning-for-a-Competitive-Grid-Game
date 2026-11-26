Training\_1 	20261029(seed)	160k(decadimento epsilon)	Riferimento(scopo)

Training\_2 	20261030(seed)	160k(decadimento epsilon)	Verificare la riproducibilità(scopo)

Training\_3	20261029(seed)	240k(decadimento epsilon)	Studiare l'effetto di ε(scopo)









**Avviare training:**

python train.py --config configs/TrainingPO_1.yaml
python train.py --config configs/TrainingPO_2.yaml



**Avviare training se si ha un file di config yaml:**
python train.py --config configs/TrainingPO_1.yaml
python train.py --config configs/TrainingPO_2.yaml



**Avviare analisi TrainingPO_1 — 16 checkpoint:**
for /L %S in (20000,20000,320000) do python evaluate.py --checkpoint runs/TrainingPO_1/checkpoint_%S.pt --device cuda --num-envs 8 --num-workers 4 --episodes-per-context 100 --eval-seed 20261015 --mode greedy --output-dir evaluations/TrainingPO_1/checkpoint_%S

**Avviare analisi TrainingPO_2 — 16 checkpoint:**
for /L %S in (20000,20000,640000) do python evaluate.py --checkpoint runs/TrainingPO_2/checkpoint_%S.pt --device cuda --num-envs 8 --num-workers 4 --episodes-per-context 100 --eval-seed 20261015 --mode greedy --output-dir evaluations/TrainingPO_2/checkpoint_%S


**Valutare la baseline Random**
python evaluate.py --checkpoint runs/TrainingPO_1/checkpoint_320000.pt --device cuda --num-envs 8 --num-workers 4 --episodes-per-context 100 --eval-seed 20261015 --mode epsilon_greedy --epsilon 1.0 --output-dir evaluations/baseline_random_pomdp


**Avviare Plots TrainingPO_1:**
python -m analysis.analyze_training --evaluations-root evaluations/TrainingPO_1 --training-metrics runs/TrainingPO_1/metrics.jsonl --baseline evaluations/baseline_random_pomdp --output-dir analysis/results/TrainingPO_1 --plots


**Avviare Plots TrainingPO_2:**
python -m analysis.analyze_training --evaluations-root evaluations/TrainingPO_2 --training-metrics runs/TrainingPO_2/metrics.jsonl --baseline evaluations/baseline_random_pomdp --output-dir analysis/results/TrainingPO_2 --plots




python train.py --config configs/Training_tot.yaml

python train.py --config configs/Training_tot.yaml --resume runs/Training_tot/checkpoint_1000000.pt --total-activations 10000000 --checkpoint-every-activations 1000000

(da lanciare dentro runs/training_X)copy /b metrics.jsonl+metrics_resume_from_1000000.jsonl metrics_combined.jsonl

for %F in (runs\Training_tot\checkpoint_*.pt) do python evaluate.py --checkpoint "%F" --device cuda --num-envs 8 --num-workers 4 --episodes-per-context 100 --eval-seed 20261015 --mode greedy --output-dir "evaluations/Training_tot/%~nF"

python evaluate.py --checkpoint runs/Training_tot/checkpoint_10000000.pt --device cuda --num-envs 8 --num-workers 4 --episodes-per-context 100 --eval-seed 20261015 --mode epsilon_greedy --epsilon 1.0 --output-dir evaluations/baseline_random

python -m analysis.analyze_training --evaluations-root evaluations/Training_tot --training-metrics runs/Training_tot/metrics_combined.jsonl --baseline evaluations/baseline_random --output-dir analysis/results/Training_tot --plots