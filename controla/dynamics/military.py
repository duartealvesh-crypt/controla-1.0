"""Force, combat de Tullock, attrition, butin (spec §3 et §9).

Deux écarts assumés par rapport à la lettre de la spec, tous deux documentés dans
le plan de build :

1. **Les apports technologiques de la Force sont saturants.** La spec écrit
   `Force = Population × (base + Guerrier + β·métal + γ·camps)`. Or le métal est
   proportionnel au nombre de cases et la population aussi, donc `Force ∝ cases²` :
   c'est exactement le scaling exponentiel que §12 interdit, parce qu'il *crée* la
   stratégie dominante. On remplace donc `β·métal` par `β·métal/(métal + m₀)`,
   borné à `β`, et idem pour les camps. La Force reste proportionnelle à la
   population — c'est voulu — mais plus quadratique en territoire.

2. **La fraction engagée est explicite** : `f_engagée = Force × ρ(Guerrier)`. Le
   trait Guerrier compte donc deux fois (multiplicateur de Force ET ardeur), ce que
   la spec veut (« deux échelles de temps ») mais qui est le premier suspect si un
   rush Guerrier domine au sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

from controla.state.game_state import Faction, GameState


def _saturating(value: float, weight: float, halflife: float) -> float:
    """`weight × value/(value + halflife)` : croît puis plafonne à `weight`.

    C'est la brique anti-exponentielle : la moitié du bonus est atteinte à
    `halflife`, et on n'atteint jamais `weight`. Des rendements décroissants,
    jamais exponentiels (§12).
    """
    if value <= 0 or halflife <= 0:
        return 0.0
    return weight * value / (value + halflife)


# ── 6. Force ────────────────────────────────────────────────────────────────

def compute_force(cfg, state: GameState, faction: Faction) -> float:
    """La Force militaire de la faction, jamais négative."""
    camps = state.building_counts(faction.id).get("camp", 0)
    quality = (
        float(cfg.force.base)
        + float(cfg.force.warrior_weight) * faction.traits["warrior"]
        + _saturating(faction.metal, float(cfg.force.metal_weight),
                      float(cfg.force.metal_halflife))
        + _saturating(camps, float(cfg.force.camp_weight),
                      float(cfg.force.camp_halflife))
    )
    return max(0.0, faction.population) * max(0.0, quality) * (1.0 - faction.hunger)


def refresh_force(cfg, state: GameState) -> dict[int, float]:
    for faction in state.living:
        faction.force = compute_force(cfg, state, faction)
    return {faction.id: faction.force for faction in state.living}


def engaged_fraction(cfg, faction: Faction) -> float:
    """`ρ(Guerrier)` : la part de sa Force qu'un peuple engage au combat.

    Un peuple très guerrier s'engage à fond, un peuple modéré envoie une petite
    troupe. Le dieu influence l'ardeur, jamais le nombre — ce serait un ordre direct.
    """
    low, high = float(cfg.force.engaged_min), float(cfg.force.engaged_max)
    span = float(cfg.traits.max) - float(cfg.traits.min)
    normalised = 0.0 if span <= 0 else (faction.traits["warrior"] - float(cfg.traits.min)) / span
    return low + (high - low) * max(0.0, min(1.0, normalised))


def engaged_force(cfg, faction: Faction) -> float:
    return faction.force * engaged_fraction(cfg, faction)


# ── 10. Combat ──────────────────────────────────────────────────────────────

@dataclass
class BattleOutcome:
    attacker: int
    defender: int
    target: tuple[int, int]
    attacker_force: float
    defender_force: float
    surprise: bool
    win_probability: float
    attacker_won: bool
    attacker_losses: float
    defender_losses: float
    annexed: tuple[int, int] | None
    loot: dict[str, float]
    allies: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {
            "attacker": self.attacker,
            "defender": self.defender,
            "target": list(self.target),
            "attacker_force": round(self.attacker_force, 2),
            "defender_force": round(self.defender_force, 2),
            "surprise": self.surprise,
            "win_probability": round(self.win_probability, 4),
            "attacker_won": self.attacker_won,
            "attacker_losses": round(self.attacker_losses, 2),
            "defender_losses": round(self.defender_losses, 2),
            "annexed": list(self.annexed) if self.annexed else None,
            "loot": {key: round(value, 2) for key, value in self.loot.items()},
            "allies": list(self.allies),
        }


def win_probability(cfg, attacker_force: float, defender_force: float,
                    surprise: bool) -> float:
    """Contest de Tullock : `(s·f_A)^m / ((s·f_A)^m + f_B^m)`.

    `m` est LE curseur chance ↔ skill. À `m=1` une armée deux fois supérieure ne
    gagne que 67 % du temps (loterie) ; à `m=2.5` elle gagne ~85 %.

    Scale-invariant : seul le ratio compte. La surprise n'est appliquée que si
    l'attaque n'a pas été détectée — elle devient ainsi la récompense d'une
    traîtrise réussie, et non un bonus permanent.
    """
    decisiveness = float(cfg.combat.decisiveness)
    effective = attacker_force * (float(cfg.combat.surprise) if surprise else 1.0)
    if effective <= 0:
        return 0.0
    if defender_force <= 0:
        return 1.0
    strong = effective ** decisiveness
    return strong / (strong + defender_force ** decisiveness)


def _attrition(cfg, base_rate: float, margin: float, stream) -> float:
    """Fraction perdue, décroissante avec la marge de victoire.

    Une victoire écrasante coûte peu, une victoire de justesse coûte cher. C'est
    **le** frein anti-snowball : un peuple qui gagne toutes ses guerres mais y
    saigne ne peut pas rouler indéfiniment sur la carte.
    """
    weight = float(cfg.combat.attrition_ratio_weight)
    rate = base_rate * (1.0 + weight * (1.0 - margin))
    rate = stream.noisy(rate, float(cfg.combat.attrition_sigma))
    return max(0.0, min(1.0, rate))


def resolve_battle(
    cfg,
    state: GameState,
    attacker: Faction,
    defender: Faction,
    target: tuple[int, int],
    surprise: bool,
    stream,
    allies: tuple[Faction, ...] = (),
) -> BattleOutcome:
    """Résout une bataille échue et applique ses conséquences.

    Si une **alliance de guerre est honorée**, les forces des alliés s'additionnent
    et le butin est partagé (§7). Les pertes sont fractionnaires et proportionnelles
    à la population, jamais un nombre absolu.
    """
    attacking = [attacker, *allies]
    attack_force = sum(engaged_force(cfg, faction) for faction in attacking)
    defend_force = engaged_force(cfg, defender)

    probability = win_probability(cfg, attack_force, defend_force, surprise)
    attacker_won = stream.random() < probability

    # La marge est vue du gagnant : proche de 1 = écrasement, proche de 0 = de justesse.
    total = attack_force + defend_force
    margin = 0.0 if total <= 0 else abs(attack_force - defend_force) / total

    winner_rate = _attrition(cfg, float(cfg.combat.attrition_winner), margin, stream)
    loser_rate = _attrition(cfg, float(cfg.combat.attrition_loser), margin, stream)

    if attacker_won:
        attacker_rate, defender_rate = winner_rate, loser_rate
    else:
        attacker_rate, defender_rate = loser_rate, winner_rate

    attacker_losses = 0.0
    for faction in attacking:
        lost = faction.population * attacker_rate
        faction.population = max(0.0, faction.population - lost)
        attacker_losses += lost
    defender_losses = defender.population * defender_rate
    defender.population = max(0.0, defender.population - defender_losses)

    annexed: tuple[int, int] | None = None
    loot: dict[str, float] = {}
    if attacker_won:
        tile = state.grid.get(*target)
        if tile.owner == defender.id:
            tile.owner = attacker.id
            tile.building = None          # la structure ne survit pas à la prise
            annexed = target

        share = float(cfg.combat.loot_fraction) * margin
        beneficiaries = attacking
        for resource in ("food", "wood", "metal"):
            taken = defender.stock(resource) * share
            if taken <= 0:
                continue
            defender.set_stock(resource, defender.stock(resource) - taken)
            per_head = taken / len(beneficiaries)
            for faction in beneficiaries:
                faction.set_stock(resource, faction.stock(resource) + per_head)
            loot[resource] = taken

    return BattleOutcome(
        attacker=attacker.id,
        defender=defender.id,
        target=target,
        attacker_force=attack_force,
        defender_force=defend_force,
        surprise=surprise,
        win_probability=probability,
        attacker_won=attacker_won,
        attacker_losses=attacker_losses,
        defender_losses=defender_losses,
        annexed=annexed,
        loot=loot,
        allies=tuple(faction.id for faction in allies),
    )
