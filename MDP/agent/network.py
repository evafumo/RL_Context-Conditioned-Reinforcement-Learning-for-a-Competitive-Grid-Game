""" Q-network CNN + MLP per Context-Conditioned Masked DQN.


Questa rete condivide i parametri fra le quattro mappe e i sei avversari.
forward() restituisce ESCLUSIVAMENTE i sette Q-value GREZZI per stato:
non applica softmax, argmax o maschere e non implementa la loss DQN.

Il masking va applicato all'esterno con environment.action_mask.mask_q_values()
solo quando si sceglie un'azione o si calcola il massimo del target Bellman.
Il Q_online(s,a) della loss si estrae dai Q-value non mascherati.

Uso su GPU (inferenza centralizzata, ambienti su CPU):
    model = QNetwork().to('cuda')
    q_values = model(spatial_batch_cuda, feature_batch_cuda)

Il modulo non inizializza CUDA all'importazione e non trasferisce dati
fra CPU e GPU all'interno di forward().
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

from environment.action_mask import NUM_ACTIONS
from environment.observations import FEATURE_SIZE, SPATIAL_SHAPE

SPATIAL_EMBEDDING_SIZE: Final[int] = 128
VECTOR_EMBEDDING_SIZE: Final[int] = 64
SHARED_EMBEDDING_SIZE: Final[int] = 128


def _init_hidden_layer(layer: nn.Module) -> None:
    """
     Descrizione:
        Inizializza i parametri dei livelli nascosti della rete neurale
        utilizzando l'inizializzazione ortogonale.

        La procedura viene applicata esclusivamente ai livelli
        Conv2d e Linear.

        I pesi vengono inizializzati mediante una matrice ortogonale
        con gain calcolato in funzione della funzione di attivazione
        ReLU, mentre i bias vengono inizializzati a zero.

        Questa strategia favorisce una propagazione stabile delle
        attivazioni e dei gradienti durante l'addestramento.

     Input:
        layer (nn.Module):
            Livello della rete neurale da inizializzare.
            Se è un'istanza di nn.Conv2d o nn.Linear, vengono
            inizializzati i relativi pesi e bias.

      Output:
        None:
            Modifica direttamente i parametri del livello ricevuto,
            senza restituire valori.
    """
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class SpatialEncoder(nn.Module):
   """
    Descrizione:
        Implementa il modulo di codifica spaziale basato su una rete
        neurale convoluzionale (CNN).

        Riceve la rappresentazione spaziale dell'ambiente, composta
        da quattro piani di dimensione 9x9, e ne estrae caratteristiche
        spaziali utili alla valutazione dello stato da parte della rete DQN.

        L'architettura comprende:
            - Tre livelli convoluzionali Conv2d con 32 canali.
            - Funzioni di attivazione ReLU dopo ogni convoluzione.
            - Un livello Flatten per convertire le mappe di attivazione
              in un vettore monodimensionale.
            - Un livello Linear che comprime la rappresentazione
              in un embedding di dimensione 128.
            - Una funzione di attivazione ReLU finale.

        Le convoluzioni utilizzano kernel 3x3, stride 1 e padding 1,
        preservando le dimensioni spaziali delle mappe durante
        l'estrazione delle caratteristiche.

        I parametri convoluzionali e lineari vengono inizializzati
        mediante la funzione _init_hidden_layer.

    Input:
        Nessuno:
            La classe non richiede parametri esterni per la propria
            costruzione. Le dimensioni dell'input sono definite
            attraverso SPATIAL_SHAPE.

    Output:
        SpatialEncoder:
            Istanza del codificatore spaziale contenente i livelli
            convoluzionali e lineari.
            
    Rappresentazione:
        Input:
            Tensor di dimensione (B, 4, 9, 9).

        Output:
            Tensor di dimensione (B, 128).

        Dove B rappresenta il numero di osservazioni nel minibatch.
  """
    def __init__(self) -> None:
        super().__init__()
        channels, height, width = SPATIAL_SHAPE
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(32 * height * width, SPATIAL_EMBEDDING_SIZE),
            nn.ReLU(),
        )
        self.apply(_init_hidden_layer)

    def forward(self, spatial: Tensor) -> Tensor:
        """Riceve (B,4,9,9) e restituisce (B,128)."""
        return self.features(spatial)


class VectorEncoder(nn.Module):
    """ 
    Descrizione:
        Implementa il modulo di codifica delle caratteristiche
        numeriche dello stato mediante una rete neurale
        completamente connessa (MLP).

        Riceve il vettore delle caratteristiche ausiliarie
        dell'osservazione e ne apprende una rappresentazione latente.

        Il vettore può contenere informazioni relative a:
            - Punteggio.
            - Tempo.
            - Turno di gioco.
            - Identificativo della mappa.
            - Informazioni relative agli avversari.

        L'architettura comprende due livelli Linear con funzione
        di attivazione ReLU.

        Il livello finale produce un embedding di dimensione 64,
        successivamente utilizzato dal modulo SharedEncoder
        per la fusione con le caratteristiche spaziali.

        I parametri lineari vengono inizializzati mediante
        la funzione _init_hidden_layer.

    Input:
        Nessuno:
            La classe utilizza FEATURE_SIZE per determinare
            la dimensione del vettore di ingresso.

    Output:
        VectorEncoder:
            Istanza del codificatore delle caratteristiche numeriche.

    Rappresentazione:
        Input:
            Tensor di dimensione (B, 13).

        Output:
            Tensor di dimensione (B, 64).
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(FEATURE_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, VECTOR_EMBEDDING_SIZE),
            nn.ReLU(),
        )
        self.apply(_init_hidden_layer)

    def forward(self, features: Tensor) -> Tensor:
        """Riceve (B,13) e restituisce (B,64)."""
        return self.features(features)


class SharedEncoder(nn.Module):
    """
     Descrizione:
        Implementa il modulo di codifica condivisa che combina
        le informazioni spaziali e numeriche dell'osservazione.

        Utilizza due codificatori indipendenti:
            - SpatialEncoder: estrae caratteristiche dalle mappe
              spaziali e produce un embedding di dimensione 128.
            - VectorEncoder: estrae caratteristiche dal vettore
              numerico e produce un embedding di dimensione 64.

        I due embedding vengono concatenati lungo la dimensione
        delle feature, ottenendo una rappresentazione complessiva
        di dimensione 192.

        La rappresentazione concatenata viene successivamente
        elaborata da una rete MLP composta da due livelli Linear
        con attivazione ReLU.

        Il risultato è un embedding condiviso di dimensione 128,
        utilizzato dalla testa Q per stimare il valore delle azioni.

        La condivisione dell'encoder consente di apprendere una
        rappresentazione comune dello stato, utilizzabile per
        valutare tutte le azioni disponibili.

     Input:
        Nessuno:
            I codificatori spaziale e numerico vengono inizializzati
            internamente alla classe.

     Output:
        SharedEncoder:
            Istanza del codificatore condiviso contenente i due
            encoder e il modulo di fusione.

     Rappresentazione:
        Input:
            spatial: (B, 4, 9, 9).
            features: (B, 13).

        Output:
            Tensor di dimensione (B, 128).
   """

    def __init__(self) -> None:
        super().__init__()
        self.spatial_encoder = SpatialEncoder()
        self.vector_encoder = VectorEncoder()
        self.fusion = nn.Sequential(
            nn.Linear(SPATIAL_EMBEDDING_SIZE + VECTOR_EMBEDDING_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, SHARED_EMBEDDING_SIZE),
            nn.ReLU(),
        )
        self.fusion.apply(_init_hidden_layer)

    def forward(self, spatial: Tensor, features: Tensor) -> Tensor:
        """Riceve (B,4,9,9) e (B,13); restituisce (B,128)."""
        spatial_embedding = self.spatial_encoder(spatial)
        vector_embedding = self.vector_encoder(features)
        return self.fusion(torch.cat((spatial_embedding, vector_embedding), dim=-1))


class QNetwork(nn.Module):
     """
    Descrizione:
        Implementa la rete neurale Q_theta(s,a) utilizzata
        dal Context-Conditioned Masked Deep Q-Network.

        L'architettura combina una CNN per l'elaborazione delle
        rappresentazioni spaziali e una MLP per l'elaborazione
        delle caratteristiche numeriche dello stato.

        La codifica delle due rappresentazioni viene effettuata
        mediante SharedEncoder, che produce un embedding condiviso
        di dimensione 128.

        L'embedding viene successivamente elaborato da una testa
        completamente connessa (Q-head), composta da un livello
        Linear che produce sette valori Q, uno per ciascuna azione
        disponibile nell'ambiente.

        L'uscita della rete rappresenta una stima dei ritorni
        attesi associati alle diverse azioni nello stato corrente.

        La testa Q utilizza un'attivazione lineare, senza applicare
        softmax, sigmoid o tanh, poiché i valori Q possono essere
        sia positivi sia negativi.

        La rete restituisce esclusivamente valori Q non mascherati.
        La gestione delle azioni illegali viene demandata al modulo
        environment.action_mask, che applica il masking durante
        la selezione delle azioni e il calcolo del massimo
        nel target Bellman.

        Il valore Q_online(s,a) utilizzato nella loss DQN deve
        essere estratto dai valori Q grezzi, senza applicare
        preventivamente la maschera.

    Input:
        Nessuno:
            I moduli interni vengono inizializzati automaticamente
            durante la costruzione della rete.

    Output:
        QNetwork:
            Istanza della rete neurale contenente l'encoder condiviso
            e la testa di valutazione delle azioni.

    Rappresentazione:
        Input:
            spatial: (B, 4, 9, 9).
            features: (B, 13).

        Output:
            Tensor di dimensione (B, 7).

        Ogni elemento del vettore di output rappresenta il valore
        Q stimato per una delle sette azioni disponibili.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SharedEncoder()
        self.q_head = nn.Linear(SHARED_EMBEDDING_SIZE, NUM_ACTIONS)
        # In DQN non si tratta di logits di policy PPO: gain=1 per la testa Q.
        nn.init.orthogonal_(self.q_head.weight, gain=1.0)
        nn.init.zeros_(self.q_head.bias)

    def _validate_inputs(self, spatial: Tensor, features: Tensor) -> None:
        """Convalida tipi, batch, geometria, dtype e device senza sync GPU."""
        if not isinstance(spatial, Tensor) or not isinstance(features, Tensor):
            raise TypeError("spatial e features devono essere tensori PyTorch")
        if (
            spatial.ndim != 4
            or tuple(spatial.shape[1:]) != SPATIAL_SHAPE
            or features.ndim != 2
            or features.shape[1] != FEATURE_SIZE
            or spatial.shape[0] != features.shape[0]
            or spatial.shape[0] < 1
        ):
            raise ValueError(
                "Attesi spatial (B,4,9,9) e features (B,13), con B >= 1"
            )
        if not spatial.is_floating_point() or not features.is_floating_point():
            raise TypeError("spatial e features devono essere floating point")
        if spatial.dtype != features.dtype:
            raise TypeError("spatial e features devono avere lo stesso dtype")
        if spatial.device != features.device:
            raise ValueError("spatial e features devono stare sullo stesso device")
        reference_parameter = self.q_head.weight
        if spatial.device != reference_parameter.device:
            raise ValueError(
                "Gli input devono stare sul device del modello: usare model.to(device) "
                "e to_torch_batch(..., device=device)"
            )
        if spatial.dtype != reference_parameter.dtype:
            raise TypeError(
                "Il dtype degli input deve coincidere con quello dei parametri "
                "(float32 consigliato, anche con autocast)"
            )

    def forward(self, spatial: Tensor, features: Tensor) -> Tensor:
        """Calcola Q non mascherati, shape (B,7), con gradienti se abilitati."""
        self._validate_inputs(spatial, features)
        embedding = self.encoder(spatial, features)
        return self.q_head(embedding)


__all__ = [
    "SpatialEncoder",
    "VectorEncoder",
    "SharedEncoder",
    "QNetwork",
    "SPATIAL_EMBEDDING_SIZE",
    "VECTOR_EMBEDDING_SIZE",
    "SHARED_EMBEDDING_SIZE",
]
