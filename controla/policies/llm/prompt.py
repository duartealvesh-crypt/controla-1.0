"""Construction des prompts, **entièrement dérivée du catalogue**.

Ce module ne contient aucun nom d'action, aucun coût, aucun libellé de bâtiment :
tout vient de `rules_text.render_rules()` et de la vue produite par l'environnement.
Un test le vérifie. C'est ce qui garantit qu'ajouter un mod le rend visible au dieu
sans que personne n'ait à toucher au prompt.

Deux parties, séparées volontairement :

* **statique** — le règlement. Identique à chaque appel d'une même partie, donc
  placé en message système et **mis en cache** par les fournisseurs qui le font.
* **dynamique** — l'état du monde et l'énumération des coups légaux.

Le dieu choisit **par index**. C'est plus robuste qu'un nom plus des coordonnées :
pas de parsing, pas de « recalage sur la case la plus proche » qui créditerait au
modèle un coup qu'il n'a pas choisi.
"""

from __future__ import annotations

from controla.registry.catalog import Catalog
from controla.rules_text import render_rules

# Schéma de sortie. Le dieu planifie TOUT son tour en un appel : c'est le « bundle »
# de §1, et c'est aussi ce qui tient le budget à ~60 appels par partie.
PLAN_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "plan_turn",
        "description": (
            "Planifie tes interventions divines pour ce tour. Choisis des coups "
            "dans la liste des actions légales, par leur index."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Ton raisonnement stratégique : pourquoi ces coups servent "
                        "ta position à long terme."
                    ),
                },
                "plan": {
                    "type": "array",
                    "description": (
                        "Les coups à jouer, dans l'ordre. Liste vide = ne rien "
                        "faire ce tour. Le coût total ne doit pas dépasser tes "
                        "points d'intervention."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "Index du coup dans la liste des actions légales.",
                            },
                            "argument": {
                                "type": "string",
                                "description": (
                                    "Argumentaire libre, uniquement pour une action "
                                    "de négociation. Ce que tu écris est lu par le "
                                    "dieu visé."
                                ),
                            },
                        },
                        "required": ["index"],
                    },
                },
            },
            "required": ["plan"],
        },
    },
}


MEMORY_NOTE = (
    "\n\n"
    "Mémoire : les tours précédents de cette partie figurent plus haut dans la "
    "conversation — l'état du monde que tu as reçu à chaque tour, et le plan que "
    "tu as rendu. Sers-t'en pour tenir une ligne, honorer ce que tu as promis et "
    "repérer ce que font tes rivaux d'un tour à l'autre."
)


def build_system(cfg, catalog: Catalog, memory: bool = False) -> str:
    """Le message système : le règlement, stable sur toute la partie.

    `memory` ajoute une phrase, et une seule, annonçant que l'historique est
    présent. Sans elle, un modèle peut lire les tours passés comme des exemples
    plutôt que comme SA partie. Elle fait partie du traitement « avec mémoire »
    de l'étude : la variante sans mémoire ne la reçoit jamais.
    """
    return (
        render_rules(cfg, catalog)
        + "\n\n"
        + "═══ TON RÔLE " + "═" * 50 + "\n"
        "Tu es l'un des dieux. À chaque tour tu reçois l'état du monde et la liste "
        "EXHAUSTIVE de tes coups légaux, chacun précédé de son index. Tu réponds "
        "par un plan : la suite d'index à jouer, dans l'ordre.\n"
        "\n"
        "Trois règles de forme :\n"
        "  • N'utilise QUE des index présents dans la liste. Un index absent est "
        "un coup perdu, compté contre toi.\n"
        "  • Le coût total de ton plan ne doit pas dépasser tes points "
        "d'intervention. Les points non dépensés sont perdus en fin de tour.\n"
        "  • Un plan vide est un choix légitime, mais rarement le meilleur.\n"
        "\n"
        "Ce qui gagne des parties ici : choisir une spécialisation adaptée à tes "
        "adversaires, s'y tenir, et dévier leurs contres par la diplomatie."
        + (MEMORY_NOTE if memory else "")
    )


def _format_traits(traits: dict, catalog: Catalog) -> str:
    return "  ".join(
        f"{catalog.label(key)} {value:+.2f}" for key, value in traits.items()
    )


def _format_self(view: dict, catalog: Catalog) -> str:
    me = view["self"]
    buildings = ", ".join(f"{catalog.label(k)}×{v}" for k, v in sorted(me["buildings"].items()))
    war = {k: v for k, v in me["war_desire"].items() if v > 0.01}
    return (
        f"─── TON PEUPLE : {me['name']} " + "─" * 30 + "\n"
        f"Population {me['population']}   Force {me['force']}   "
        f"faim {me['hunger']:.0%}   ardeur au combat {me['engaged_fraction']:.0%}\n"
        f"Nourriture {me['food']}   Bois {me['wood']}   Métal {me['metal']}\n"
        f"Recherche {me['research']} (palier {me['research_tier']})   "
        f"Territoire {me['tiles']} cases   Bâtiments : {buildings or 'aucun'}\n"
        f"Traits : {_format_traits(me['traits'], catalog)}\n"
        f"Envie de guerre de ton peuple : {war or 'aucune'}\n"
        f"POINTS D'INTERVENTION DISPONIBLES : {me['intervention_points']}"
    )


def _format_rivals(view: dict, catalog: Catalog) -> str:
    blocks = []
    for rival in view["rivals"]:
        if not rival["alive"]:
            blocks.append(
                f"  {rival['name']} — ÉLIMINÉ au tour {rival['eliminated_turn']}"
            )
            continue
        reputation = rival["reputation"]
        parole = (
            f"promesses tenues {reputation['honored']}/{reputation['total']}"
            if reputation["total"] else "ne s'est jamais engagé"
        )
        buildings = sum(rival["buildings"].values())
        line = (
            f"  {rival['name']} — {rival['tiles']} cases, {buildings} bâtiments"
            f"{'  ⚔ EN GUERRE CONTRE TOI' if rival['at_war_with_me'] else ''}\n"
            f"      relation commerciale {rival['trade_relation']:+.2f}   ·   {parole}"
        )
        if "revealed" in rival:
            revealed = rival["revealed"]
            line += (
                f"\n      [omniscience] pop {revealed['population']}  "
                f"force {revealed['force']}  recherche {revealed['research']}  "
                f"nour {revealed['food']}  bois {revealed['wood']}  "
                f"mét {revealed['metal']}"
                f"\n      [omniscience] traits : "
                f"{_format_traits(revealed['traits'], catalog)}"
            )
        blocks.append(line)
    return "\n".join(blocks) or "  (aucun rival)"


def _format_legal_actions(view: dict, catalog: Catalog, limit: int) -> str:
    """Énumère les coups légaux, groupés par action.

    `limit` borne le nombre de cibles listées par action : sur une grille de 121
    cases, une bénédiction a des dizaines de cibles et le prompt exploserait. La
    troncature est **annoncée** — un silence ferait croire au dieu qu'il a tout vu.
    """
    grouped: dict[str, list[dict]] = {}
    for item in view["legal_actions"]:
        grouped.setdefault(item["action"], []).append(item)

    blocks = []
    for action_key, items in grouped.items():
        cost = items[0]["cost"]
        shown = items[:limit]
        entries = "  ".join(
            f"[{item['index']}] "
            + (item["label"].split("→", 1)[1].strip() if "→" in item["label"] else "—")
            for item in shown
        )
        header = f"{action_key} ({cost} pt{'s' if cost > 1 else ''})"
        if len(items) > len(shown):
            header += f"  — {len(shown)} cibles listées sur {len(items)}"
        blocks.append(f"{header}\n    {entries}")
    return "\n".join(blocks)


def build_user(cfg, catalog: Catalog, view: dict, retry_note: str | None = None) -> str:
    """Le message utilisateur : l'état du tour et les coups jouables."""
    turns_left = view["max_turns"] - view["turn"]
    pending = view["pending"]
    pending_text = (
        "\n".join(
            f"  • {effect['kind']} au tour {effect['due_turn']} — {effect['cause']}"
            for effect in pending
        )
        or "  (rien en cours)"
    )
    correction = f"\n[CORRECTION] {retry_note}\n" if retry_note else ""

    inbox = "\n".join(
        f"  • {message['from_name']} te propose « {message['pledge']} » :\n"
        f'    "{message["argument"]}"' if message["argument"]
        else f"  • {message['from_name']} te propose « {message['pledge']} » (sans mot)"
        for message in view.get("inbox", [])
    ) or "  (aucun message)"

    commitments = "\n".join(
        f"  • {item['kind']} avec {item['partner_name']}, jusqu'au tour "
        f"{item['expires_turn']}"
        + ("  ⚠ ROMPU" if item["breached_turn"] is not None else
           f"  (honoré {len(item['honoured_turns'])} fois)")
        for item in view.get("commitments", [])
    ) or "  (aucun engagement)"

    return (
        f"═══ TOUR {view['turn']}/{view['max_turns']} "
        f"— {turns_left} tours restants " + "═" * 20 + "\n\n"
        f"{_format_self(view, catalog)}\n\n"
        "─── RIVAUX " + "─" * 45 + "\n"
        f"{_format_rivals(view, catalog)}\n\n"
        "─── MESSAGES REÇUS " + "─" * 37 + "\n"
        f"{inbox}\n\n"
        "─── TES ENGAGEMENTS " + "─" * 36 + "\n"
        f"{commitments}\n\n"
        "─── TES ACTIONS EN COURS " + "─" * 31 + "\n"
        f"{pending_text}\n\n"
        "─── COUPS LÉGAUX " + "─" * 39 + "\n"
        f"{_format_legal_actions(view, catalog, int(cfg.llm.max_legal_listed))}\n"
        f"{correction}\n"
        "Planifie ton tour."
    )
