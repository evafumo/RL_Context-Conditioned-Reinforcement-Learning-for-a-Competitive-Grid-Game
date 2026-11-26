"""Visualizzatore Pygame delle partite DQN.

Uso:
    python -m pip install pygame
    python pygame_viewer.py
    python pygame_viewer.py --device cuda:0
    python pygame_viewer.py --checkpoints-dir runs/Training_tot --seed 42

Il motore di gioco e' SOLO environment.game.GridGame: questo modulo non
implementa ne' modifica le regole dell'MDP. Le policy Opponent continuano a
essere quelle originali e la policy Agent e' greedy (epsilon=0).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

try:
    import pygame
except ImportError as exc:
    raise SystemExit("Pygame mancante. Installa con: python -m pip install pygame") from exc

import torch

from agent.action_selection import select_actions
from environment.action_mask import legal_action_mask
from environment.game import GameState, GridGame, HORIZON, StepResult
from environment.observations import encode_observation
from environment.rules import (
    GRID_SIZE,
    Action,
    MOVEMENT_ACTIONS,
    Player,
    ShotOutcome,
    MeleeOutcome,
    move_destination,
    other_player,
    respawn_positions,
)
from evaluate import load_evaluation_network

# I checkpoint devono trovarsi nella cartella originale (non inclusi nel .zip).
DIFFICULTIES: Final[dict[str, tuple[str, int]]] = {
    "Facile": ("checkpoint_100000.pt", 100_000),
    "Medio": ("checkpoint_800000.pt", 800_000),
    "Difficile": ("checkpoint_4000000.pt", 4_000_000),
}

WIDTH, HEIGHT = 1330, 820
CELL = 64
BOARD_X, BOARD_Y = 75, 151
BOARD_SIDE = GRID_SIZE * CELL
PANEL_X, PANEL_W = 725, 530
FPS = 60
FLASH_MS = 170  # 6 semiperiodi -> esattamente 3 lampeggi bianchi
SPEEDS_MS = (1300, 850, 540, 300, 150)
SPEED_LABELS = ("Molto lenta", "Lenta", "Normale", "Veloce", "Molto veloce")

BG = (15, 22, 35)
CARD = (28, 41, 59)
CARD_BORDER = (58, 77, 99)
TEXT = (235, 242, 250)
MUTED = (160, 178, 199)
GREEN = (74, 215, 133)
RED = (237, 94, 104)
GOLD = (255, 203, 93)
WHITE = (255, 255, 255)
NAVY = (18, 29, 45)

OPPONENT_NAMES: Final[dict[int, str]] = {
    1: "Random", 2: "Aggressive", 3: "Hunter",
    4: "Hybrid", 5: "Shooter", 6: "ASH",
}
ACTION_NAMES: Final[dict[Action, str]] = {
    Action.NORTH: "Nord", Action.SOUTH: "Sud", Action.EAST: "Est",
    Action.WEST: "Ovest", Action.SHOOT: "Tiro", Action.PASS: "Pass",
    Action.DUMMY: "Azione forzata",
}


def player_name(player: Player | None) -> str:
    return {Player.AGENT: "Agent", Player.OPPONENT: "Opponent", None: "Nessuno"}[player]


@dataclass(frozen=True)
class BoardView:
    """Snapshot utilizzabile anche quando l'MDP entra nello stato terminale."""

    agent_position: tuple[int, int]
    opponent_position: tuple[int, int]
    objectives: frozenset[tuple[int, int]]
    agent_vp: int
    opponent_vp: int

    @classmethod
    def from_state(cls, state: GameState) -> "BoardView":
        return cls(
            state.agent_position, state.opponent_position,
            state.active_objectives, state.agent_vp, state.opponent_vp,
        )

    @property
    def differential(self) -> int:
        return self.agent_vp - self.opponent_vp


def _post_move_positions(result: StepResult) -> tuple[tuple[int, int], tuple[int, int]]:
    """Ricostruisce SOLO le coordinate visive prima di un eventuale respawn."""
    a, o = result.state.agent_position, result.state.opponent_position
    action = result.events.effective_action
    if action in MOVEMENT_ACTIONS:
        if result.state.player_to_act is Player.AGENT:
            a = move_destination(a, action)
        else:
            o = move_destination(o, action)
    return a, o


def _post_scores(result: StepResult) -> tuple[int, int]:
    if result.outcome is not None:
        return result.outcome.agent_vp, result.outcome.opponent_vp
    if isinstance(result.next_state, GameState):
        return result.next_state.agent_vp, result.next_state.opponent_vp
    raise AssertionError("StepResult senza punteggio finale")


def combat_board(result: StepResult) -> BoardView:
    """Fotogramma DOPO l'azione ma PRIMA del respawn per il lampeggio."""
    a, o = _post_move_positions(result)
    agent_vp, opponent_vp = _post_scores(result)
    # In caso di rigenerazione, i nuovi obiettivi sono visibili quando
    # il simulatore ha restituito uno stato non terminale.
    if isinstance(result.next_state, GameState):
        objectives = result.next_state.active_objectives
    else:
        objectives = result.state.active_objectives - frozenset(
            (result.events.captured_objective,)
            if result.events.captured_objective is not None else ()
        )
    return BoardView(a, o, frozenset(objectives), agent_vp, opponent_vp)


def board_after_step(result: StepResult) -> BoardView:
    """Stato visualizzato al termine della transizione, anche al turno 100.

    Lo stato s-dagger e' astratto: lo snapshot finale ricostruisce la posizione
    dell'ultima mossa. Se l'ultimo obiettivo rigenera 3 nuovi obiettivi al
    turno 100, le nuove celle non sono esposte dal simulatore terminale;
    in quel raro caso la schermata finale le omette intenzionalmente.
    """
    if isinstance(result.next_state, GameState):
        return BoardView.from_state(result.next_state)
    before_respawn = combat_board(result)
    if result.events.killer is not None:
        a, o = respawn_positions()
        return BoardView(a, o, before_respawn.objectives,
                         before_respawn.agent_vp, before_respawn.opponent_vp)
    return before_respawn


def describe_step(result: StepResult) -> str:
    ev = result.events
    actor = player_name(result.state.player_to_act)
    parts: list[str] = []
    if ev.effective_action is None:
        parts.append(f"{actor}: corpo a corpo obbligatorio")
    else:
        parts.append(f"{actor}: {ACTION_NAMES[ev.effective_action]}")
    if ev.captured_objective is not None:
        parts.append("obiettivo +1")
    if ev.objectives_regenerated:
        parts.append("nuovi obiettivi")
    if ev.shot_outcome is ShotOutcome.MISS:
        parts.append("colpo mancato")
    if ev.shot_outcome is ShotOutcome.TARGET_KILLED:
        parts.append("colpo letale")
    if ev.melee_outcome is MeleeOutcome.BOTH_SURVIVE:
        parts.append("entrambi sopravvivono")
    if ev.melee_outcome is MeleeOutcome.ATTACKER_KILLS:
        parts.append("attaccante vincitore")
    if ev.melee_outcome is MeleeOutcome.DEFENDER_KILLS:
        parts.append("contrattacco letale")
    if ev.killer is not None:
        parts.append(f"{player_name(ev.killer)} elimina {player_name(other_player(ev.killer))}")
    return "  |  ".join(parts)


class Match:
    """Collega una singola policy DQN greedy al motore MDP e alle animazioni."""

    def __init__(
        self, difficulty: str, model: torch.nn.Module,
        device: torch.device, *, seed: int | None = None,
    ) -> None:
        self.difficulty = difficulty
        self.model = model
        self.device = device
        # Lo shaping influenza solo il reward RL, non le transizioni o i VP.
        self.game = GridGame(seed=seed, shaping_lambda=0.0)
        first = self.game.reset()  # mappa/opponent/inizialiva casuali
        self.initiative = first.player_to_act
        self.display = BoardView.from_state(first)
        self.flash_view: BoardView | None = None
        self.flash_victim: Player | None = None
        self.flash_elapsed_ms = 0
        self.next_view: BoardView | None = None
        self.displayed_turn = 0
        self.last_actor: Player | None = None
        self.last_event = "Partita iniziata."
        self.paused = False
        self.speed_index = 2
        self.step_elapsed_ms = 0

    @property
    def interval_ms(self) -> int:
        return SPEEDS_MS[self.speed_index]

    @property
    def finished(self) -> bool:
        return self.game.terminated and self.flash_view is None

    @property
    def current_actor(self) -> Player | None:
        if self.game.terminated:
            return None
        state = self.game.state
        assert isinstance(state, GameState)
        return state.player_to_act

    def select_agent_action(self) -> Action:
        legal = self.game.legal_actions()
        if legal == (Action.DUMMY,):
            return Action.DUMMY
        state = self.game.state
        assert isinstance(state, GameState)
        obs = encode_observation(state, self.game.context)
        mask = legal_action_mask(self.game)
        spatial = torch.from_numpy(obs.spatial).unsqueeze(0).to(self.device)
        features = torch.from_numpy(obs.features).unsqueeze(0).to(self.device)
        action_mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            action_id = int(select_actions(
                self.model, spatial, features, action_mask, 0.0,
            ).item())
        selected = Action(action_id)
        if selected not in legal:
            raise RuntimeError("Il DQN ha scelto un'azione illegale")
        return selected

    def advance(self) -> StepResult:
        """Una e una sola attivazione, senza calcolare regole nel renderer."""
        if self.game.terminated or self.flash_view is not None:
            raise RuntimeError("Avanzamento non consentito: fine partita o animazione")
        action = self.select_agent_action()
        result = self.game.step(action)
        self.displayed_turn = result.state.activation
        self.last_actor = result.state.player_to_act
        self.last_event = describe_step(result)
        self.next_view = board_after_step(result)
        if result.events.killer is not None:
            # Il modello applica immediatamente il respawn: la GUI mostra
            # prima un'istantanea della posizione della vittima eliminata.
            self.flash_view = combat_board(result)
            self.flash_victim = other_player(result.events.killer)
            self.flash_elapsed_ms = 0
            self.display = self.flash_view
        else:
            self.display = self.next_view
            self.next_view = None
        return result

    def update(self, dt_ms: int) -> None:
        if self.paused or self.finished:
            return
        if self.flash_view is not None:
            self.flash_elapsed_ms += dt_ms
            if self.flash_elapsed_ms >= FLASH_MS * 6:
                assert self.next_view is not None
                self.display = self.next_view  # adesso appare il respawn
                self.flash_view = None
                self.flash_victim = None
                self.next_view = None
                self.flash_elapsed_ms = 0
                self.step_elapsed_ms = 0
            return
        self.step_elapsed_ms += dt_ms
        if self.step_elapsed_ms >= self.interval_ms:
            self.step_elapsed_ms = 0
            self.advance()

    def flashing_white(self, player: Player) -> bool:
        return (
            self.flash_view is not None
            and self.flash_victim is player
            and (self.flash_elapsed_ms // FLASH_MS) % 2 == 0
        )

    def result_message(self) -> str:
        outcome = self.game.outcome
        if outcome is None:
            raise RuntimeError("La partita non e' ancora terminata")
        if outcome.winner is None:
            return "Pareggio: differenziale 0"
        # Differenziale della prospettiva del vincitore, sempre positivo.
        return (
            f"Ha vinto {player_name(outcome.winner)} "
            f"con differenziale {abs(outcome.score_differential)}"
        )


def draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str,
              pos: tuple[int, int], color: tuple[int, int, int] = TEXT,
              *, center: bool = False) -> pygame.Rect:
    img = font.render(str(text), True, color)
    rect = img.get_rect(center=pos) if center else img.get_rect(topleft=pos)
    surface.blit(img, rect)
    return rect


def card(surface: pygame.Surface, rect: pygame.Rect) -> None:
    pygame.draw.rect(surface, CARD, rect, border_radius=15)
    pygame.draw.rect(surface, CARD_BORDER, rect, width=1, border_radius=15)


def button(surface: pygame.Surface, fonts: dict[str, pygame.font.Font],
           rect: pygame.Rect, title: str, subtitle: str = "",
           *, accent: tuple[int, int, int] = GREEN,
           hovered: bool = False) -> None:
    fill = (51, 71, 92) if hovered else CARD
    pygame.draw.rect(surface, fill, rect, border_radius=13)
    pygame.draw.rect(surface, accent if hovered else CARD_BORDER, rect,
                     width=2 if hovered else 1, border_radius=13)
    draw_text(surface, fonts["normal"], title,
              (rect.centerx, rect.centery - (11 if subtitle else 0)), accent, center=True)
    if subtitle:
        draw_text(surface, fonts["small"], subtitle,
                  (rect.centerx, rect.centery + 20), MUTED, center=True)


class ViewerApp:
    def __init__(self, checkpoints_dir: Path, device: torch.device,
                 seed: int | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("Giulio2 | DQN Grid Arena")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.fonts = {
            "tiny": pygame.font.SysFont("segoeui", 18),
            "small": pygame.font.SysFont("segoeui", 21),
            "normal": pygame.font.SysFont("segoeui", 28, bold=True),
            "large": pygame.font.SysFont("segoeui", 36, bold=True),
            "hero": pygame.font.SysFont("segoeui", 56, bold=True),
        }
        self.checkpoints_dir = checkpoints_dir
        self.device = device
        self.seed = seed
        self.match: Match | None = None
        self.error: str | None = None
        self.models: dict[str, torch.nn.Module] = {}
        self.running = True
        self.menu_buttons = {
            difficulty: pygame.Rect(465, 298 + i * 104, 400, 83)
            for i, difficulty in enumerate(DIFFICULTIES)
        }
        self.end_again = pygame.Rect(381, 505, 255, 62)
        self.end_menu = pygame.Rect(694, 505, 255, 62)

    def start_match(self, difficulty: str) -> None:
        path = self.checkpoints_dir / DIFFICULTIES[difficulty][0]
        try:
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint non trovato: {path}")
            if difficulty not in self.models:
                network, metadata = load_evaluation_network(path, self.device)
                steps = metadata.get("_checkpoint_activations")
                expected = DIFFICULTIES[difficulty][1]
                if steps != expected:
                    raise ValueError(
                        f"Contatore checkpoint: {steps}, atteso: {expected}"
                    )
                self.models[difficulty] = network
            self.match = Match(difficulty, self.models[difficulty],
                               self.device, seed=self.seed)
            self.error = None
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.match = None

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            return
        if self.match is None:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                for index, difficulty in enumerate(DIFFICULTIES, start=pygame.K_1):
                    if event.key == index:
                        self.start_match(difficulty)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for difficulty, rect in self.menu_buttons.items():
                    if rect.collidepoint(event.pos):
                        self.start_match(difficulty)
            return

        match = self.match
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.match = None
            elif event.key == pygame.K_SPACE and not match.finished:
                match.paused = not match.paused
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                match.speed_index = min(match.speed_index + 1, len(SPEEDS_MS) - 1)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                match.speed_index = max(match.speed_index - 1, 0)
            elif event.key == pygame.K_r and match.finished:
                self.start_match(match.difficulty)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and match.finished:
            if self.end_again.collidepoint(event.pos):
                self.start_match(match.difficulty)
            elif self.end_menu.collidepoint(event.pos):
                self.match = None

    def draw_menu(self) -> None:
        screen = self.screen
        fonts = self.fonts
        draw_text(screen, fonts["hero"], "GRID ARENA",
                  (WIDTH // 2, 139), center=True)
        draw_text(screen, fonts["small"], "Il DQN gioca contro un avversario automatico",
                  (WIDTH // 2, 199), MUTED, center=True)
        draw_text(screen, fonts["small"], "Scegli lo stadio di addestramento",
                  (WIDTH // 2, 238), MUTED, center=True)
        mouse = pygame.mouse.get_pos()
        for i, (difficulty, rect) in enumerate(self.menu_buttons.items(), start=1):
            filename, activations = DIFFICULTIES[difficulty]
            accent = {"Facile": GREEN, "Medio": GOLD, "Difficile": RED}[difficulty]
            button(screen, fonts, rect, f"{i}  -  {difficulty.upper()}",
                   f"Checkpoint {activations:,} attivazioni".replace(",", "."),
                   accent=accent, hovered=rect.collidepoint(mouse))
        draw_text(screen, fonts["tiny"], "Mappa casuale  |  Opponent casuale  |  100 attivazioni",
                  (WIDTH // 2, 654), MUTED, center=True)
        draw_text(screen, fonts["tiny"], "ESC per uscire",
                  (WIDTH // 2, 690), MUTED, center=True)
        if self.error:
            # Mostra il messaggio, senza interrompere la finestra Pygame.
            message = self.error
            if len(message) > 116:
                message = "..." + message[-113:]
            draw_text(screen, fonts["tiny"], message, (WIDTH // 2, 757), RED, center=True)

    def square_rect(self, pos: tuple[int, int]) -> pygame.Rect:
        x, y = pos
        return pygame.Rect(
            BOARD_X + (x - 1) * CELL,
            BOARD_Y + (GRID_SIZE - y) * CELL,
            CELL, CELL,
        )

    def draw_board(self, match: Match) -> None:
        pygame.draw.rect(self.screen, (37, 52, 71),
                         pygame.Rect(BOARD_X - 5, BOARD_Y - 5,
                                     BOARD_SIDE + 10, BOARD_SIDE + 10),
                         border_radius=10)
        obstacles = match.game.context.game_map.obstacles
        goals = match.display.objectives
        for x in range(1, GRID_SIZE + 1):
            for y in range(1, GRID_SIZE + 1):
                cell = self.square_rect((x, y))
                color = (31, 49, 66) if (x + y) % 2 == 0 else (36, 56, 75)
                pygame.draw.rect(self.screen, color, cell)
                pygame.draw.rect(self.screen, (56, 76, 96), cell, width=1)
                if (x, y) in obstacles:
                    pygame.draw.rect(self.screen, (91, 107, 123),
                                     cell.inflate(-8, -8), border_radius=7)
                    pygame.draw.line(self.screen, (116, 132, 145),
                                     (cell.left + 11, cell.top + 11),
                                     (cell.right - 11, cell.bottom - 11), 2)
                if (x, y) in goals:
                    pygame.draw.circle(self.screen, (171, 120, 42), cell.center, 16)
                    pygame.draw.circle(self.screen, GOLD, cell.center, 12)
                    pygame.draw.circle(self.screen, WHITE, (cell.centerx - 3, cell.centery - 3), 3)

        for position, player, normal, letter in (
            (match.display.agent_position, Player.AGENT, GREEN, "A"),
            (match.display.opponent_position, Player.OPPONENT, RED, "O"),
        ):
            rect = self.square_rect(position).inflate(-14, -14)
            pygame.draw.rect(self.screen,
                             WHITE if match.flashing_white(player) else normal,
                             rect, border_radius=8)
            draw_text(self.screen, self.fonts["normal"], letter, rect.center, NAVY, center=True)

        for i in range(1, GRID_SIZE + 1):
            cx = BOARD_X + (i - 0.5) * CELL
            cy = BOARD_Y + (GRID_SIZE - i + 0.5) * CELL
            draw_text(self.screen, self.fonts["tiny"], str(i),
                      (int(cx), BOARD_Y + BOARD_SIDE + 15), MUTED, center=True)
            draw_text(self.screen, self.fonts["tiny"], str(i),
                      (BOARD_X - 21, int(cy)), MUTED, center=True)

    def draw_panel(self, match: Match) -> None:
        font = self.fonts
        screen = self.screen
        context = match.game.context
        draw_text(screen, font["large"], "Partita in corso", (PANEL_X, 77))

        r1 = pygame.Rect(PANEL_X, 148, PANEL_W, 130)
        card(screen, r1)
        draw_text(screen, font["small"], "DIFFICOLTÀ", (PANEL_X + 21, 161), MUTED)
        draw_text(screen, font["normal"], match.difficulty, (PANEL_X + 21, 187), GOLD)
        draw_text(screen, font["small"], f"Opponent: {OPPONENT_NAMES[int(context.opponent)]}",
                  (PANEL_X + 220, 166))
        draw_text(screen, font["small"], f"Mappa: {context.map_id}",
                  (PANEL_X + 220, 203), MUTED)
        draw_text(screen, font["tiny"], "Context-conditioned DQN  |  policy greedy (epsilon = 0)",
                  (PANEL_X + 21, 246), MUTED)

        r2 = pygame.Rect(PANEL_X, 292, PANEL_W, 139)
        card(screen, r2)
        draw_text(screen, font["small"], "INIZIATIVA INIZIALE", (PANEL_X + 21, 307), MUTED)
        draw_text(screen, font["normal"], player_name(match.initiative),
                  (PANEL_X + 21, 333), TEXT)
        actor = match.current_actor
        who = player_name(actor) if actor is not None else "Partita terminata"
        draw_text(screen, font["small"], "PROSSIMO A GIOCARE", (PANEL_X + 270, 307), MUTED)
        draw_text(screen, font["normal"], who,
                  (PANEL_X + 270, 333), GREEN if actor is Player.AGENT else RED if actor is Player.OPPONENT else TEXT)
        draw_text(screen, font["small"], f"Attivazione completata: {match.displayed_turn}/{HORIZON}",
                  (PANEL_X + 21, 388), MUTED)

        r3 = pygame.Rect(PANEL_X, 445, PANEL_W, 164)
        card(screen, r3)
        draw_text(screen, font["small"], "PUNTEGGIO", (PANEL_X + 21, 460), MUTED)
        draw_text(screen, font["normal"], "Agent", (PANEL_X + 24, 492), GREEN)
        draw_text(screen, font["normal"], f"{match.display.agent_vp:+d}",
                  (PANEL_X + 173, 492), TEXT)
        draw_text(screen, font["normal"], "Opponent", (PANEL_X + 287, 492), RED)
        draw_text(screen, font["normal"], f"{match.display.opponent_vp:+d}",
                  (PANEL_X + 460, 492), TEXT)
        pygame.draw.line(screen, CARD_BORDER, (PANEL_X + 22, 544),
                         (PANEL_X + PANEL_W - 22, 544), 1)
        draw_text(screen, font["small"], "Differenziale D = VP Agent - VP Opponent",
                  (PANEL_X + 22, 558), MUTED)
        draw_text(screen, font["normal"], f"{match.display.differential:+d}",
                  (PANEL_X + 448, 557), TEXT)

        r4 = pygame.Rect(PANEL_X, 624, PANEL_W, 126)
        card(screen, r4)
        draw_text(screen, font["small"], "ULTIMA ATTIVAZIONE", (PANEL_X + 21, 637), MUTED)
        event_line = match.last_event
        # Spezza in due righe sui separatori degli eventi, senza troncamenti strani.
        if len(event_line) > 44:
            pieces = event_line.split("  |  ")
            lines = ["", ""]
            for piece in pieces:
                idx = 0 if len(lines[0]) < 42 else 1
                lines[idx] = (lines[idx] + "  |  " if lines[idx] else "") + piece
        else:
            lines = [event_line]
        for index, line in enumerate(lines[:2]):
            if len(line) > 56:
                line = line[:53] + "..."
            draw_text(screen, font["tiny"], line,
                      (PANEL_X + 21, 671 + index * 30), TEXT)
        status = "PAUSA" if match.paused else "Riproduzione automatica"
        draw_text(screen, font["tiny"],
                  f"{status}   |   Velocità: {SPEED_LABELS[match.speed_index]}",
                  (PANEL_X, 767), GOLD if match.paused else MUTED)

    def draw_end(self, match: Match) -> None:
        shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        shade.fill((7, 12, 22, 205))
        self.screen.blit(shade, (0, 0))
        rect = pygame.Rect(245, 233, 840, 369)
        card(self.screen, rect)
        draw_text(self.screen, self.fonts["large"], "PARTITA TERMINATA",
                  (WIDTH // 2, 292), GOLD, center=True)
        draw_text(self.screen, self.fonts["normal"], match.result_message(),
                  (WIDTH // 2, 365), TEXT, center=True)
        draw_text(self.screen, self.fonts["small"],
                  f"VP Agent: {match.display.agent_vp}    |    VP Opponent: {match.display.opponent_vp}"
                  f"    |    D = {match.display.differential:+d}",
                  (WIDTH // 2, 423), MUTED, center=True)
        mouse = pygame.mouse.get_pos()
        button(self.screen, self.fonts, self.end_again, "Rigioca",
               hovered=self.end_again.collidepoint(mouse))
        button(self.screen, self.fonts, self.end_menu, "Menu",
               accent=GOLD, hovered=self.end_menu.collidepoint(mouse))

    def draw_game(self, match: Match) -> None:
        draw_text(self.screen, self.fonts["large"], "ARENA 9 x 9", (BOARD_X, 71))
        draw_text(self.screen, self.fonts["tiny"], "OSTACOLI  |  OBIETTIVI  |  COMBATTIMENTO AUTOMATICO",
                  (BOARD_X, 114), MUTED)
        self.draw_board(match)
        self.draw_panel(match)
        draw_text(self.screen, self.fonts["tiny"],
                  "SPAZIO pausa   |   +/- velocità   |   ESC menu",
                  (BOARD_X, 767), MUTED)
        if match.finished:
            self.draw_end(match)

    def run(self) -> None:
        try:
            while self.running:
                dt_ms = self.clock.tick(FPS)
                for event in pygame.event.get():
                    self.handle_event(event)
                if self.match is not None:
                    try:
                        self.match.update(dt_ms)
                    except (RuntimeError, ValueError, AssertionError) as exc:
                        self.error = f"Errore durante la partita: {exc}"
                        self.match = None
                self.screen.fill(BG)
                if self.match is None:
                    self.draw_menu()
                else:
                    self.draw_game(self.match)
                pygame.display.flip()
        finally:
            pygame.quit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualizzatore Pygame DQN Giulio2")
    parser.add_argument("--checkpoints-dir", type=Path,
                        default=Path(__file__).resolve().parent / "runs" / "Training_tot")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0")
    parser.add_argument("--seed", type=int, default=None,
                        help="Opzionale: ripete la stessa mappa, opponent e RNG")
    args = parser.parse_args(argv)
    device_name = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("Dispositivo CUDA non disponibile: usare --device cpu")
    app = ViewerApp(args.checkpoints_dir.expanduser(), device, seed=args.seed)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
