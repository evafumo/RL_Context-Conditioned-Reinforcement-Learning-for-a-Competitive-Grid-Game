
"""Agente DQN del progetto Giulio2.

Questo package contiene i componenti necessari per implementare
il Context-Conditioned Masked Deep Q-Network (DQN standard).

Componenti previsti:
    network:
        Q-network CNN + MLP condizionata al contesto.

    action_selection:
        Selezione epsilon-greedy con action masking.

    replay_buffer:
        Experience replay uniforme.

    dqn:
        Algoritmo DQN standard con target network.

    parallel_env:
        Simulazione di ambienti paralleli su CPU.

    collector:
        Raccolta delle transizioni per l'addestramento.

Il package non contiene le regole del gioco.
La legalità delle azioni è determinata dal simulatore MDP.

I moduli vengono importati esplicitamente per evitare
dipendenze circolari ed effetti collaterali.

L'inizializzazione di PyTorch e dei dispositivi CUDA è
delegata ai moduli che eseguono effettivamente i calcoli.
"""
