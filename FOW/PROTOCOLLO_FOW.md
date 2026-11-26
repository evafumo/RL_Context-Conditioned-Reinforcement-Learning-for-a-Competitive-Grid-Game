# Protocollo FOW/Belief-DQN compatibile con MDP originale

Installazione: **sostituire** i file con i relativi percorsi nella cartella `FOW` esistente. Non cancellare cartelle `runs/` e `evaluations/` gia' generate. Dopo l'aggiornamento sono disponibili le metriche **A/B/C** e la valutazione a **24 contesti × 100 episodi**, come nell'MDP originale.

## Training (da cartella FOW, CMD / Anaconda Prompt)

Prima fase (1.000.000 attivazioni):

```cmd
python train.py --config configs/Training_tot.yaml
```

Seconda fase (ripresa da 1.000.000 fino a 10.000.000 attivazioni cumulative):

```cmd
python train.py --config configs/Training_tot.yaml --resume runs/Training_tot/checkpoint_1000000.pt --total-activations 10000000 --checkpoint-every-activations 1000000
```

Il YAML incluso abilita `fog_of_war: true`, `fog_radius: 4`, `belief_filter: true`. Contiene i medesimi iperparametri della configurazione fornita, incluso `epsilon.decay_activations: 8000000`. Attenzione a non sovrascrivere la configurazione dopo avere creato il checkpoint iniziale: la firma serve al resume.

## Valutazione di tutti i checkpoint, baseline e analisi

Terminato il training, su Windows CMD / Anaconda Prompt:

```cmd
run_protocollo_Training_tot.cmd
```

Questo script:

1. Unisce `metrics.jsonl` e `metrics_resume_from_1000000.jsonl` in `metrics_combined.jsonl` (valido per **esattamente due** sessioni di training, senza ulteriori resume).
2. Valuta tutti i checkpoint disponibili a policy **greedy** con **belief** e raggio Manhattan 4, con **100 episodi per ciascuno dei 24 contesti**, `--eval-seed 20261015`. Ogni cartella `evaluations/Training_tot/checkpoint_N/` conterra' `summary.json` e `episodes.jsonl`.
3. Calcola una baseline Random con `--mode epsilon_greedy --epsilon 1.0` sui **medesimi seed/contesti**: e' uniforme sulle azioni legali, non usa i Q-value del checkpoint; il checkpoint serve per il caricamento tecnico della rete.
4. Lancia `analysis.analyze_training`, esportando statistiche A/B/C, CSV, JSON e grafici.

Le valutazioni sono eseguite dopo il training, senza modificarne i checkpoint. L'utilita' primaria resta `U=1{differenziale finale > 0}`: i pareggi sono conteggiati tra le non-vittorie.

## Comandi manuali

```cmd
python evaluate.py --checkpoint runs/Training_tot/checkpoint_1000000.pt --device cuda:0 --num-envs 8 --num-workers 4 --episodes-per-context 100 --eval-seed 20261015 --mode greedy --observation belief --fog-radius 4 --output-dir evaluations/Training_tot/checkpoint_1000000

python evaluate.py --checkpoint runs/Training_tot/checkpoint_10000000.pt --device cuda:0 --num-envs 8 --num-workers 4 --episodes-per-context 100 --eval-seed 20261015 --mode epsilon_greedy --epsilon 1.0 --observation belief --fog-radius 4 --output-dir evaluations/baseline_random

python -m analysis.analyze_training --evaluations-root evaluations/Training_tot --training-metrics runs/Training_tot/metrics_combined.jsonl --baseline evaluations/baseline_random --output-dir analysis/results/Training_tot --plots
```

Il vecchio comando FOW con `--mode belief --episodes 2400 --seed 42 --json evaluations/belief/results.json` rimane utilizzabile: ora produce **anche** `summary.json` e `episodes.jsonl`, con campionamento bilanciato (100 episodi per contesto).

Per un test breve, aggiungere `--contexts 1:1,1:2 --episodes-per-context 2` alla valutazione e usare una diversa cartella di output; non mescolare i risultati smoke con quelli completi.

**Modalita' disponibili:** `--mode greedy/epsilon_greedy` specifica la politica, `--observation full/mask/belief/qmdp` specifica l'osservazione. Le forme brevi `--mode full/mask/belief/qmdp` sono equivalenti a `--mode greedy --observation ...`. Per `qmdp` e `full` la rete deve essere stata addestrata senza Fog; per `belief` deve essere addestrata con `fog_of_war + belief_filter`; per `mask` con Fog senza belief. Il checkpoint di baseline Random puo' avere qualsiasi tipo di osservazione, dato che epsilon=1 non usa le sue predizioni.

**Confronti:** il sistema di analisi verifica i seed episodio per episodio e impedisce il confronto di esperimenti con raggi o modalita' di osservazione differenti, oltre ai vincoli originali su orizzonte, shaping e dimensione dei campioni. Se il training e' gia' iniziato con la vecchia versione priva di diagnostiche B, tali statistiche non potranno essere recuperate retroattivamente; i nuovi log invece includono TD-MAE, TD-P90, deviazioni dei Q e massimi Q legali, senza modificare loss o algoritmo DQN.

Per la suite ridotta di regressione:

```cmd
python -m pytest tests/test_fow_evaluation_protocol.py -q
```
