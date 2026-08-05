"""Espace d'action plat, index statique, masque dynamique.

Le patron est celui d'une arène RL classique : **l'espace d'action a une taille
fixe**, connue au `reset`, et la légalité est portée entièrement par un masque
booléen recalculé à chaque observation. C'est ce qui permet à une policy RL de
poser une tête de sortie de taille constante, et à une policy LLM de recevoir une
énumération exhaustive des coups jouables (§11.4 : jamais de repli aléatoire).

L'index est construit une fois : pour chaque action du catalogue, on énumère son
**domaine statique** de cibles (toutes les cases, ou toutes les factions, ou rien).
Le domaine ne dépend pas de l'état — seul le masque en dépend.

Le « bundle » d'actions de la spec §1 devient la séquence de coups joués entre deux
`end_turn` : un coup par `step()`, ce qui garde l'espace plat et maskable.
"""

from __future__ import annotations

from dataclasses import dataclass

from controla.registry.catalog import Catalog
from controla.registry.descriptors import (
    COMMITMENT_KINDS,
    WAR_ALLIANCE,
    ActionKind,
    ActionSpec,
    TargetKind,
)
from controla.state.game_state import GameState

# Cible : coordonnée de case, id de faction, (id de faction, engagement), ou rien.
# Cible : coordonnée de case, id de faction, triplet de négociation, ou rien.
# Le triplet est (partenaire, engagement, cible_de_guerre) ; `cible_de_guerre`
# vaut NO_WAR_TARGET pour tout engagement qui n'en réclame pas.
Target = tuple[int, int] | tuple[int, str, int] | int | None

NO_WAR_TARGET = -1


@dataclass(frozen=True)
class ActionEntry:
    """Une entrée de l'index plat : un couple (action, cible) jouable en principe."""

    index: int
    spec: ActionSpec
    target: Target

    @property
    def key(self) -> str:
        return self.spec.key

    def describe(self, state: GameState | None = None) -> str:
        """Libellé court destiné à l'énumération dans le prompt."""
        if self.target is None:
            return self.spec.key
        if isinstance(self.target, tuple) and len(self.target) == 3:
            partner, commitment, victim = self.target
            name = state.faction(partner).name if state else partner
            if commitment == COMMITMENT_KINDS[0]:
                return f"{self.spec.key} → {name}"
            if victim != NO_WAR_TARGET:
                victim_name = state.faction(victim).name if state else victim
                return f"{self.spec.key} → {name} [{commitment} contre {victim_name}]"
            return f"{self.spec.key} → {name} [{commitment}]"
        if isinstance(self.target, int):
            name = state.faction(self.target).name if state else self.target
            return f"{self.spec.key} → {name}"
        return f"{self.spec.key} → ({self.target[0]},{self.target[1]})"


class ActionIndex:
    """L'index plat des actions. Construit une fois par partie."""

    __slots__ = ("entries", "_by_pair", "end_turn_index", "_catalog")

    def __init__(self, catalog: Catalog, grid_size: int, num_factions: int):
        self._catalog = catalog
        entries: list[ActionEntry] = []
        for spec in catalog.actions:
            for target in self._domain(spec, grid_size, num_factions):
                entries.append(ActionEntry(len(entries), spec, target))
        self.entries: tuple[ActionEntry, ...] = tuple(entries)
        self._by_pair = {(entry.key, entry.target): entry.index for entry in self.entries}
        self.end_turn_index = next(
            entry.index for entry in self.entries
            if entry.spec.kind is ActionKind.CONTROL
        )

    @staticmethod
    def _domain(spec: ActionSpec, grid_size: int, num_factions: int) -> list[Target]:
        """Domaine STATIQUE des cibles. Indépendant de l'état — c'est le masque qui
        décide de la légalité du moment.

        `OWN_TILE` et `ENEMY_TILE` partagent le même domaine (toutes les cases) :
        qui possède quoi change en cours de partie, donc ça relève du masque.
        """
        if spec.target_kind is TargetKind.NONE:
            return [None]
        if spec.target_kind in (TargetKind.OWN_TILE, TargetKind.ENEMY_TILE):
            return [(x, y) for x in range(grid_size) for y in range(grid_size)]
        if spec.target_kind is TargetKind.FACTION:
            return list(range(num_factions))
        if spec.target_kind is TargetKind.FACTION_COMMITMENT:
            domain: list[Target] = []
            for partner in range(num_factions):
                for commitment in COMMITMENT_KINDS:
                    if commitment == WAR_ALLIANCE:
                        # §7 : « attaquer la MÊME CIBLE ». Nommer la cible dans la
                        # promesse la rend symétrique — sans elle, quand un allié
                        # appelle à la guerre et pas l'autre, le moteur devrait
                        # blâmer arbitrairement le silencieux, qui n'avait pourtant
                        # promis de suivre aucune cible en particulier.
                        domain.extend(
                            (partner, commitment, victim)
                            for victim in range(num_factions)
                        )
                    else:
                        domain.append((partner, commitment, NO_WAR_TARGET))
            return domain
        raise ValueError(f"TargetKind non géré : {spec.target_kind}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> ActionEntry:
        return self.entries[index]

    def find(self, action_key: str, target: Target = None) -> int:
        """Index d'un couple (action, cible). Lève si le couple n'existe pas.

        Les cibles de case arrivent parfois en liste (JSON) : on normalise.
        """
        if isinstance(target, list) and len(target) == 2:
            target = (int(target[0]), int(target[1]))
        try:
            return self._by_pair[(action_key, target)]
        except KeyError:
            raise KeyError(
                f"couple hors de l'espace d'action : {action_key!r} / {target!r}"
            ) from None


# ── légalité ────────────────────────────────────────────────────────────────

def is_legal(cfg, state: GameState, faction_id: int, entry: ActionEntry) -> bool:
    """Une action est-elle jouable ici et maintenant ?

    SOURCE UNIQUE de légalité : le masque, le prompt et l'exécution passent tous
    par cette fonction. Un agent ne peut donc pas proposer une action illégale par
    construction.
    """
    faction = state.faction(faction_id)
    if not faction.alive:
        return False

    # `end_turn` est toujours jouable : sans elle, un dieu sans point pourrait
    # bloquer la partie.
    if entry.spec.kind is ActionKind.CONTROL:
        return True

    if faction.intervention_points < entry.spec.cost:
        return False

    kind = entry.spec.target_kind
    if kind is TargetKind.NONE:
        return True

    if kind is TargetKind.FACTION:
        target = state.faction(entry.target)          # type: ignore[arg-type]
        return target.id != faction_id and target.alive

    if kind is TargetKind.FACTION_COMMITMENT:
        # L'engagement étant replié dans l'index, promettre une alliance à un mort
        # — ou une guerre contre soi-même — est simplement hors masque.
        partner_id, commitment, victim = entry.target   # type: ignore[misc]
        partner = state.faction(partner_id)
        if partner.id == faction_id or not partner.alive:
            return False
        if commitment != WAR_ALLIANCE:
            return victim == NO_WAR_TARGET
        return (
            victim not in (faction_id, partner_id)
            and 0 <= victim < len(state.factions)
            and state.faction(victim).alive
        )

    tile = state.grid.get(*entry.target)              # type: ignore[misc]
    if kind is TargetKind.OWN_TILE:
        return tile.owner == faction_id
    if kind is TargetKind.ENEMY_TILE:
        return tile.owner is not None and tile.owner != faction_id
    raise ValueError(f"TargetKind non géré : {kind}")


def legal_mask(cfg, state: GameState, faction_id: int, index: ActionIndex) -> list[int]:
    """Masque booléen sur tout l'espace d'action, dans l'ordre de l'index."""
    return [
        1 if is_legal(cfg, state, faction_id, entry) else 0
        for entry in index.entries
    ]


def legal_entries(
    cfg, state: GameState, faction_id: int, index: ActionIndex
) -> list[ActionEntry]:
    """Les actions légales, pour l'énumération dans le prompt."""
    return [
        entry for entry in index.entries
        if is_legal(cfg, state, faction_id, entry)
    ]
