from types import SimpleNamespace

from environment.fog_of_war import apply_fog_of_war
from environment.game import GridGame
from environment.observations import SpatialChannel, encode_observation, stack_observations

g = GridGame(seed=1)
s = g.reset()
obs = stack_observations([encode_observation(s, g.context)])
OPP = SpatialChannel.OPPONENT

# 1) Partenza: agente (5,1), opponent (5,9) -> distanza 8 -> nascosto con raggio 3
out = apply_fog_of_war(obs.spatial, [s], 3)
assert out[0, OPP].sum() == 0, "Opponent lontano dovrebbe essere nascosto"
assert obs.spatial[0, OPP].sum() == 1, "L'array originale non deve essere modificato"

# 2) Confini del raggio, con stati finti (conta solo la distanza Manhattan)
def fake(agent, opponent):
    return SimpleNamespace(agent_position=agent, opponent_position=opponent)

visible_3 = apply_fog_of_war(obs.spatial, [fake((5, 1), (5, 4))], 3)  # distanza 3
hidden_4 = apply_fog_of_war(obs.spatial, [fake((5, 1), (5, 5))], 3)   # distanza 4
assert visible_3[0, OPP].sum() == 1, "A distanza 3 deve restare visibile"
assert hidden_4[0, OPP].sum() == 0, "A distanza 4 deve essere nascosto"

# 3) Gli altri piani non cambiano
for ch in (SpatialChannel.OBSTACLES, SpatialChannel.AGENT, SpatialChannel.OBJECTIVES):
    assert (out[0, ch] == obs.spatial[0, ch]).all()

# 4) Raggio troppo piccolo rifiutato (sotto la portata di tiro = 3)
try:
    apply_fog_of_war(obs.spatial, [s], 2)
    raise AssertionError("Un raggio < 3 doveva essere rifiutato")
except ValueError:
    pass

print("OK: fog of war funziona correttamente")
