@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Eseguire questo script solo DOPO il training completo da 10 milioni.
if not exist "runs\Training_tot\checkpoint_10000000.pt" (
    echo ERRORE: checkpoint finale non trovato. Completare prima il training.
    exit /b 1
)
if not exist "runs\Training_tot\metrics.jsonl" (
    echo ERRORE: log prima fase mancante.
    exit /b 1
)
if not exist "runs\Training_tot\metrics_resume_from_1000000.jsonl" (
    echo ERRORE: log seconda fase mancante.
    exit /b 1
)

rem Combina i log delle due fasi nel corretto ordine cronologico.
copy /b "runs\Training_tot\metrics.jsonl"+"runs\Training_tot\metrics_resume_from_1000000.jsonl" "runs\Training_tot\metrics_combined.jsonl" >nul
if errorlevel 1 exit /b 1

rem 24 contesti x 100 episodi; metrica greedy, osservazione belief.
if not exist "evaluations\Training_tot" mkdir "evaluations\Training_tot"
for %%F in ("runs\Training_tot\checkpoint_*.pt") do (
    echo Valutazione %%~nxF
    python evaluate.py --checkpoint "%%~fF" --device cuda:0 --num-envs 8 --num-workers 4 --episodes-per-context 100 --eval-seed 20261015 --mode greedy --observation belief --fog-radius 4 --output-dir "evaluations\Training_tot\%%~nF"
    if errorlevel 1 exit /b 1
)

rem Baseline Random: stessa distribuzione, stesso seed e stessa visibilita'.
python evaluate.py --checkpoint "runs\Training_tot\checkpoint_10000000.pt" --device cuda:0 --num-envs 8 --num-workers 4 --episodes-per-context 100 --eval-seed 20261015 --mode epsilon_greedy --epsilon 1.0 --observation belief --fog-radius 4 --output-dir "evaluations\baseline_random"
if errorlevel 1 exit /b 1

rem Analisi A / B / C e grafici.
python -m analysis.analyze_training --evaluations-root "evaluations\Training_tot" --training-metrics "runs\Training_tot\metrics_combined.jsonl" --baseline "evaluations\baseline_random" --output-dir "analysis\results\Training_tot" --plots
if errorlevel 1 exit /b 1

echo Protocollo completo. Risultati in analysis\results\Training_tot
endlocal
