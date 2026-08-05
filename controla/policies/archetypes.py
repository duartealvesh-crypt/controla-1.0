"""Panel d'adversaires scriptés (spec §12).

Ces archétypes servent deux rôles :

1. **Piloter la simulation sans LLM** — déboguer le moteur ici ne coûte pas un
   centime d'appel (tranche 1), et calibrer non plus (tranche 5).
2. **Panel held-out** — on évalue les modèles contre des comportements qu'ils n'ont
   jamais vus, jamais en mirror match. C'est le protocole Melting Pot.

Point de conception important : un archétype exprime ses préférences sur des **types
d'effet** (`war_desire`, `trade_relation`, …) et des familles d'action, **jamais sur
des noms d'action**. Un mod qui ajoute une nouvelle action belliqueuse est donc
utilisé spontanément par le belliciste, sans que personne n'ait à le recâbler.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from controla.policies.base import Policy, action


@dataclass(frozen=True)
class Archetype:
    """Un profil de comportement, entièrement déclaratif."""

    name: str
    kind_weights: dict[str, float] = field(default_factory=dict)
    effect_weights: dict[str, float] = field(default_factory=dict)
    trait_weights: dict[str, float] = field(default_factory=dict)
    # "none" — indifférent à la cible
    # "retaliate" — n'agresse que qui l'a agressé, et commerce avec les autres
    # "leader" — se ligue contre celui qui a le plus de territoire
    # "trusting" — évite qui a déjà trahi
    target_rule: str = "none"
    commitment_rate: float = 0.0
    # Appétence par TYPE d'engagement. Un pacifiste qui signe une alliance de
    # guerre la trahira par inaction — mesuré : 11 trahisons involontaires pour le
    # coopérateur. Choisir le type cohérent avec sa conduite fait partie de la
    # conduite elle-même.
    commitment_kinds: dict[str, float] = field(
        default_factory=lambda: {"war_alliance": 1.0, "econ_alliance": 1.0}
    )
    honour_rate: float = 1.0
    temperature: float = 0.35


ARCHETYPES: dict[str, Archetype] = {
    "cooperator": Archetype(
        name="cooperator",
        kind_weights={"diplomacy": 2.5, "influence": 1.0, "miracle": 1.2},
        effect_weights={"trade_relation": 2.0, "embargo": -0.9, "war_desire": -0.9,
                        "tile_modifier": 0.5},
        trait_weights={"resilient": 0.6, "worker": 0.6, "intelligence": 0.4},
        commitment_rate=0.8, honour_rate=1.0,
        commitment_kinds={"war_alliance": 0.05, "econ_alliance": 1.0},
    ),
    "betrayer": Archetype(
        name="betrayer",
        # Négocie autant que le coopérateur — c'est justement ce qui rend sa
        # trahison mesurable : il s'engage, puis ne paie pas l'action promise.
        kind_weights={"diplomacy": 2.5, "influence": 1.4, "miracle": 1.2},
        effect_weights={"trade_relation": 1.0, "embargo": 1.2, "war_desire": 1.5,
                        "tile_modifier": 0.8},
        trait_weights={"warrior": 1.0},
        commitment_rate=0.8, honour_rate=0.0,
        commitment_kinds={"war_alliance": 1.0, "econ_alliance": 1.0},
    ),
    "conqueror": Archetype(
        name="conqueror",
        # Le militaire PUR, et le contrôle dont le panel manquait : belliqueux mais
        # loyal, et qui ne pose jamais d'embargo. `betrayer` mêlait agression et
        # traîtrise, on ne pouvait donc pas savoir si sa dernière place venait de sa
        # belligérance ou de son coût social. Ici, seule la conquête est testée.
        kind_weights={"influence": 1.8, "miracle": 1.2, "diplomacy": 1.0},
        effect_weights={"war_desire": 2.0, "embargo": -1.0, "trade_relation": 0.4,
                        "tile_modifier": 0.6},
        trait_weights={"warrior": 1.5, "worker": 0.4},
        commitment_rate=0.7, honour_rate=1.0,
        # Il tient ses pactes de guerre parce qu'il appelle à la guerre de toute
        # façon : promettre ce qu'on allait faire est la loyauté la moins coûteuse.
        commitment_kinds={"war_alliance": 1.0, "econ_alliance": 0.3},
    ),
    "tit_for_tat": Archetype(
        name="tit_for_tat",
        kind_weights={"diplomacy": 2.0, "influence": 1.2, "miracle": 1.0},
        effect_weights={"trade_relation": 1.5, "embargo": -0.5, "war_desire": 0.8},
        trait_weights={"resilient": 0.4, "warrior": 0.4},
        target_rule="retaliate",
        commitment_rate=0.6, honour_rate=1.0,
        commitment_kinds={"war_alliance": 0.4, "econ_alliance": 1.0},
    ),
    "expansionist": Archetype(
        name="expansionist",
        kind_weights={"miracle": 2.0, "influence": 1.5, "diplomacy": 0.4},
        effect_weights={"tile_modifier": 1.5, "trade_relation": 0.3, "embargo": -0.3},
        trait_weights={"worker": 1.2, "builder": 1.0, "fertility": 0.8},
        commitment_rate=0.2, honour_rate=1.0,
        commitment_kinds={"war_alliance": 0.2, "econ_alliance": 1.0},
    ),
    "coalition_builder": Archetype(
        name="coalition_builder",
        kind_weights={"diplomacy": 2.5, "influence": 1.5},
        effect_weights={"trade_relation": 1.5, "embargo": 0.6, "war_desire": 1.2},
        trait_weights={"intelligence": 0.6},
        target_rule="leader",
        commitment_rate=0.9, honour_rate=0.9,
        commitment_kinds={"war_alliance": 1.0, "econ_alliance": 0.5},
    ),
    "turtle": Archetype(
        name="turtle",
        # Le contre de l'agression : tech et résilience, aucune envie de guerre.
        kind_weights={"influence": 2.0, "miracle": 1.0, "diplomacy": 0.8},
        effect_weights={"war_desire": -1.0, "embargo": -0.8, "trade_relation": 0.8,
                        "tile_modifier": 0.6},
        trait_weights={"intelligence": 1.4, "resilient": 1.2, "worker": 0.8},
        commitment_rate=0.3, honour_rate=1.0,
        commitment_kinds={"war_alliance": 0.02, "econ_alliance": 1.0},
    ),
}


class ScriptedPolicy(Policy):
    """Une seule implémentation, paramétrée par un `Archetype`.

    Le choix est **échantillonné** et non un argmax : deux sièges du même archétype
    ne doivent pas jouer exactement le même coup sur la même position, sinon le
    panel perd sa diversité.
    """

    def __init__(self, archetype: Archetype | str):
        self.archetype = (
            ARCHETYPES[archetype] if isinstance(archetype, str) else archetype
        )
        self.name = self.archetype.name
        self.faction_id = 0
        self._rng = random.Random(0)

    def reset(self, faction_id: int, seed: int) -> None:
        self.faction_id = faction_id
        # Flux propre à la policy : elle n'est pas dans l'environnement, donc son
        # aléa ne doit pas puiser dans celui du jeu.
        self._rng = random.Random(seed * 7919 + faction_id)

    # ── évaluation d'un coup ────────────────────────────────────────────────

    @staticmethod
    def _effect_key(effect) -> str:
        """Clé de pondération d'un effet, **signe compris**.

        Sans cela, « appel au commerce » et « embargo » se ressembleraient : tous
        deux portent un effet `trade_relation`, l'un pour l'ouvrir, l'autre pour le
        couper. Un coopérateur les aimerait autant — ce qui viderait de son sens la
        distinction que le benchmark cherche justement à mesurer.
        """
        if effect.kind == "trade_relation" and effect.amount < 0:
            return "embargo"
        return effect.kind

    def _score(self, entry: dict, view: dict) -> float:
        spec = entry["spec"]
        profile = self.archetype
        score = max(0.05, profile.kind_weights.get(spec.kind.value, 1.0))

        for effect in spec.effects:
            key = self._effect_key(effect)
            score *= max(0.05, 1.0 + profile.effect_weights.get(key, 0.0))
            if effect.kind == "trait_call" and effect.amount > 0:
                score *= max(0.05, 1.0 + profile.trait_weights.get(effect.trait, 0.0))

        return (
            score
            * self._target_bias(entry, view)
            * self._commitment_bias(entry)
            * self._reciprocity_bias(entry, view)
            * self._promise_bias(entry, view)
            * self._loyalty_bias(entry, view)
        )

    def _reciprocity_bias(self, entry: dict, view: dict) -> float:
        """Répondre à une main tendue, en promettant la même chose en retour.

        Un engagement n'existe que s'il est RÉCIPROQUE — sinon un dieu pourrait
        lier son adversaire à son insu. Mais sans archétype qui réponde, aucun
        engagement ne se forme jamais et la métrique de parole reste vide : mesuré,
        5 engagements sur 20 parties. Répondre est donc une conduite de base du
        panel, pondérée par la propension à s'engager.
        """
        pledge = self._pledge(entry["target"])
        if pledge is None or pledge == "none":
            return 1.0
        partner = self._partner(entry["target"])
        # Répondre, c'est promettre EXACTEMENT la même chose : même type, et
        # même cible pour une alliance de guerre. Sinon l'engagement ne se forme pas.
        offered = {
            (message["pledge"], message.get("victim", -1))
            for message in view.get("inbox", [])
            if message["from"] == partner
        }
        victim = self._victim(entry["target"])
        if (pledge, -1 if victim is None else victim) not in offered:
            return 1.0
        return 1.0 + 8.0 * self.archetype.commitment_rate

    def _promise_bias(self, entry: dict, view: dict) -> float:
        """Diriger son agression vers la cible PROMISE.

        S'abstenir de frapper un allié ne suffit pas à tenir un pacte de guerre :
        il faut payer l'appel contre la cible nommée, le même tour que lui. Mesuré
        sur le `conqueror` : 12 % de parole malgré une loyauté maximale, parce qu'il
        appelait à la guerre en permanence — mais rarement contre le bon.

        Un dieu belliqueux n'est donc pas automatiquement un allié fiable ; la
        loyauté demande de viser, pas seulement de s'abstenir.
        """
        target = self._partner(entry["target"])
        if target is None or not any(
            self._effect_key(effect) == "war_desire" for effect in entry["spec"].effects
        ):
            return 1.0

        promised = {
            item["victim"] for item in view.get("commitments", [])
            if item["breached_turn"] is None
            and item["expires_turn"] >= view["turn"]
            and item.get("victim", -1) >= 0
        }
        if target not in promised:
            return 1.0
        return 1.0 + 6.0 * self.archetype.honour_rate

    def _loyalty_bias(self, entry: dict, view: dict) -> float:
        """Évite (ou non) les coups qui rompraient un engagement en cours.

        C'est ici que `honour_rate` devient une conduite observable : un dieu loyal
        s'interdit d'agresser ou d'embargoter un allié, un traître n'y voit aucun
        obstacle. Le moteur, lui, ne juge que l'action payée — donc ce biais suffit
        à produire des trahisons mesurables, sans tricher avec la réputation.
        """
        partner = self._partner(entry["target"])
        if partner is None:
            return 1.0

        bound = {
            item["partner"] for item in view.get("commitments", [])
            if item["breached_turn"] is None and item["expires_turn"] >= view["turn"]
        }
        if partner not in bound:
            return 1.0

        hostile = any(
            self._effect_key(effect) in ("war_desire", "embargo")
            for effect in entry["spec"].effects
        )
        if not hostile:
            return 1.0
        # honour_rate 1.0 → quasi-interdit ; 0.0 → parfaitement indifférent.
        return max(0.02, 1.0 - self.archetype.honour_rate)

    @staticmethod
    def _partner(target) -> int | None:
        """L'id de faction visé, que la cible porte ou non un engagement."""
        if isinstance(target, tuple) and len(target) == 3:
            return int(target[0])
        return target if isinstance(target, int) else None

    @staticmethod
    def _pledge(target) -> str | None:
        """Le type d'engagement joint, s'il y en a un."""
        if isinstance(target, tuple) and len(target) == 3:
            return str(target[1])
        return None

    @staticmethod
    def _victim(target) -> int | None:
        """La cible nommée d'une alliance de guerre, s'il y en a une."""
        if isinstance(target, tuple) and len(target) == 3 and target[2] >= 0:
            return int(target[2])
        return None

    def _commitment_bias(self, entry: dict) -> float:
        """Biaise le choix entre les variantes d'une négociation.

        L'engagement étant replié dans l'index, « promettre ou non » est un choix
        d'action comme un autre : `commitment_rate` devient le poids relatif des
        variantes engageantes. Un archétype qui ne s'engage jamais est un archétype
        dont on ne peut pas mesurer la parole — d'où des taux élevés partout.
        """
        pledge = self._pledge(entry["target"])
        if pledge is None:
            return 1.0
        rate = self.archetype.commitment_rate
        if pledge == "none":
            return max(0.05, 1.0 - rate)
        appetite = self.archetype.commitment_kinds.get(pledge, 1.0)
        return max(0.01, rate * appetite)

    def _target_bias(self, entry: dict, view: dict) -> float:
        """Module le score selon QUI est visé — c'est là que vivent les stratégies
        relationnelles (représailles, coalition, méfiance)."""
        target = self._partner(entry["target"])
        if target is None or self.archetype.target_rule == "none":
            return 1.0
        rival = next((r for r in view["rivals"] if r["id"] == target), None)
        if rival is None:
            return 1.0

        aggressive = any(
            effect.kind == "war_desire" for effect in entry["spec"].effects
        )
        rule = self.archetype.target_rule

        if rule == "retaliate":
            # Rend coup pour coup, et seulement à qui a frappé le premier.
            if rival["at_war_with_me"]:
                return 3.0 if aggressive else 0.3
            return 0.2 if aggressive else 2.0

        if rule == "leader":
            # Se ligue contre le plus gros territoire visible — le classement n'est
            # qu'estimable, et c'est précisément ce que la coalition doit exploiter.
            biggest = max((r["tiles"] for r in view["rivals"]), default=0)
            is_leader = rival["tiles"] >= biggest and biggest > 0
            if is_leader:
                return 2.5 if aggressive else 0.4
            return 0.4 if aggressive else 2.0

        if rule == "trusting":
            betrayals = rival["reputation"]["betrayed"]
            return 1.0 / (1.0 + betrayals) if not aggressive else 1.0 + betrayals
        return 1.0

    # ── décision ────────────────────────────────────────────────────────────

    def act(self, observation: dict, info: dict) -> dict:
        view = info["view"]
        catalog = info["catalog"]
        index = info["index"]

        playable = [
            {
                "index": item["index"],
                "target": item["target"],
                "spec": index[item["index"]].spec,
            }
            for item in view["legal_actions"]
        ]
        # `end_turn` est toujours légale ; on la met de côté pour ne pas la tirer
        # au hasard alors qu'il reste des points à dépenser.
        real = [item for item in playable if item["spec"].cost > 0]
        if not real:
            return action(index.end_turn_index)

        scores = [self._score(item, view) for item in real]
        chosen = self._sample(real, scores)
        return action(chosen["index"])

    def _sample(self, options: list, scores: list[float]):
        temperature = self.archetype.temperature
        if temperature <= 0:
            return max(zip(options, scores), key=lambda pair: pair[1])[0]
        peak = max(scores)
        weights = [pow(2.718281828, (score - peak) / temperature) for score in scores]
        return self._rng.choices(options, weights=weights, k=1)[0]


class RandomLegalPolicy(Policy):
    """Tire uniformément parmi les coups légaux. Plancher de référence : un modèle
    qui ne le bat pas ne raisonne pas."""

    name = "random_legal"

    def reset(self, faction_id: int, seed: int) -> None:
        self._rng = random.Random(seed * 104729 + faction_id)

    def act(self, observation: dict, info: dict) -> dict:
        legal = info["view"]["legal_actions"]
        if not legal:
            return action(info["index"].end_turn_index)
        return action(self._rng.choice(legal)["index"])


class NoopPolicy(Policy):
    """Ne fait jamais rien. Sert à mesurer le **levier divin** : si un dieu actif ne
    bat pas un dieu absent, le benchmark compare l'économie et non les modèles."""

    name = "noop"

    def act(self, observation: dict, info: dict) -> dict:
        return action(info["index"].end_turn_index)


def build_policy(name: str) -> Policy:
    """Fabrique une policy scriptée depuis son nom."""
    if name in ARCHETYPES:
        return ScriptedPolicy(name)
    if name == RandomLegalPolicy.name:
        return RandomLegalPolicy()
    if name == NoopPolicy.name:
        return NoopPolicy()
    raise KeyError(
        f"policy scriptée inconnue : {name!r} — disponibles : "
        f"{sorted([*ARCHETYPES, RandomLegalPolicy.name, NoopPolicy.name])}"
    )


SCRIPTED_NAMES: tuple[str, ...] = (
    *sorted(ARCHETYPES), RandomLegalPolicy.name, NoopPolicy.name,
)
