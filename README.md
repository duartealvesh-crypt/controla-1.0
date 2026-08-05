# Controla

Arène **god-4X pour benchmark de LLM** : quatre dieux, chacun piloté par un modèle,
règnent sur un peuple semi-autonome et s'affrontent sur une petite carte.

Ce qu'on mesure : la **cohérence stratégique long-horizon**, le raisonnement sous
incertitude, et surtout la **théorie de l'esprit** — négocier, anticiper, honorer ou
trahir. Le projet est un banc d'essai, pas un jeu commercial.

L'arène est une vraie arène de reinforcement learning : `pettingzoo.utils.env.AECEnv`,
espaces `gymnasium`, espace d'action plat masqué, reward, seeding auditable. Un agent
RL, une policy scriptée et un LLM s'y branchent par la même interface.

La spec canonique est [`docs/DESIGN.md`](docs/DESIGN.md). Les écarts assumés et les
mesures d'équilibrage sont dans [`docs/FINDINGS.md`](docs/FINDINGS.md).

---

## Les deux principes qui portent tout

**1. Le dieu agit DIRECTEMENT sur le monde, INDIRECTEMENT sur son peuple.**
Ses miracles frappent les cases. Sur son peuple, il ne donne **aucun ordre** : il
incline des *traits*, et le peuple agit de lui-même dans la passe de développement.

**2. Le peuple est non-agentique.** Aucun autopilote, aucune seconde IA. Ce qui
ressemble à « le peuple agit » n'est que des règles plus l'aléa de la passe. Mais
*indirect ≠ aléatoire* : les règles driver → comportement sont déterministes et
lisibles. On cache les **coefficients**, jamais les **causes**.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Le règlement complet, **dérivé des seuls descripteurs** — c'est aussi la partie
statique du prompt LLM :

```bash
python -m controla.cli describe
```

Une partie entre archétypes scriptés, avec son replay :

```bash
python -m controla.cli play --seed 42 --policies cooperator,betrayer,expansionist,turtle
```

Stocker une partie puis la rejouer :

```bash
python -m controla.cli play --seed 42 --out runs/game.json --journal
```

```bash
python -m controla.cli replay runs/game.json
```

## Boucle de jeu

```
reset(seed)                        → monde symétrique, traits neutres
pour chaque tour (15 par défaut) :
    pour chaque dieu, à tour de rôle :
        obs = observe(dieu)        → publique + information_state privé + masque
        step(action)               → bufferisé en secret, aucune fuite
        ... jusqu'à end_turn
    ── barrière de ronde ──
    les actions de la ronde s'appliquent ensemble
    PASSE DE DÉVELOPPEMENT         → 14 étapes, ordre figé
fin : score pondéré → classement complet des 4 → OpenSkill
```

Le « bundle » d'actions est la **séquence de coups entre deux `end_turn`**. Rien n'est
appliqué avant la barrière : un dieu qui joue en dernier ne voit pas les coups des
autres, donc la résolution est réellement simultanée.

## Architecture

```
controla/
├── config/           constantes [TUNE] en YAML, immuables au chargement
├── registry/         descripteurs déclaratifs + catalogue (la couche moddable)
├── world/            grille, BFS, génération symétrique par rotation
├── state/            factions, file d'effets datés, modificateurs
├── rng.py            aléa keyé (seed, tour, purpose, faction) + journal auditable
├── dynamics/         la passe : économie, émergence, militaire, dérive, score
├── actions/          index plat + masque (légalité) · exécution par descripteur
├── env/              ControlaEnv (AEC), observation avec brouillard, reward
├── policies/         interface · archétypes scriptés · (LLM en tranche 2)
├── arena/            play_match · (tournoi et OpenSkill en tranche 4)
├── replay/           rendu texte d'un état et d'une partie
└── cli.py            describe · play · replay
```

### Moddable par construction

Chaque élément — ressource, terrain, trait, bâtiment, palier, action — est un
**descripteur déclaratif** que le moteur exécute et que le prompt auto-décrit. Les
effets sont de la **donnée introspectable** (`EffectSpec(kind="trait_call",
trait="warrior", amount=2.0)`), jamais des lambdas opaques : sinon on peut les
*exécuter* mais pas les *auto-décrire*, et « moddable » devient un vœu.

Ajouter un mod = ajouter un descripteur dans `registry/catalog.py` et ses constantes
dans `config/default.yaml`. Il apparaît alors **automatiquement** dans le règlement,
dans le prompt et dans le masque d'actions légales.

Trois tests verrouillent cette discipline : tout descripteur doit se décrire, aucun
nom d'action ni libellé ne peut vivre en dur hors du catalogue, et aucune constante
numérique ne peut apparaître dans le catalogue.

### Aléa keyé — l'équité du benchmark

Les flux d'aléa sont dérivés de `(seed, tour, purpose, faction)` via `blake2b`,
**jamais** de l'ordre d'appel et **jamais** de `hash()` (dont le hachage des `str` est
salé par processus, ce qui produirait un non-déterminisme silencieux entre deux
exécutions).

Sans cela, un modèle qui joue plus d'actions décalerait les récoltes de ses voisins,
et le classement mesurerait la chance. Un test le prouve : ajouter trois coups à un
dieu ne change **rien** pour les autres.

## Tests

```bash
pytest
```

222 tests, ~45 s. Le filet est conçu pour rester vert pendant l'équilibrage : les
invariants sont indépendants des constantes, donc un sweep qui les casse a cassé le
moteur et non l'équilibre.

| Fichier | Ce qu'il garantit |
|---|---|
| `test_env_api.py` | conformité PettingZoo (`api_test`, `seed_test`), masque, bufferisation, coups invalides |
| `test_invariants.py` | stocks ≥ 0, traits bornés, guerre symétrique, terminaison, **conservation des ressources** |
| `test_determinism.py` | même seed ⇒ même trajectoire d'aléa · l'activité d'un dieu ne décale pas les autres |
| `test_dynamics.py` | Force saturante, Tullock, attrition, envies, dérive, économie |
| `test_registry.py` | auto-description, ordre de la passe épinglé, zéro duplication |
| `test_world.py` | symétrie de rotation **exacte** du monde, distances BFS |
| `test_llm_policy.py` | prompt dérivé du catalogue, validation, relances, métriques |
| `test_diplomacy.py` | formation des engagements, vérification sur l'intention, réputation |

## État d'avancement

- [x] **T0** — squelette et registre déclaratif
- [x] **T1** — simulation déterministe, sans LLM
- [x] **T2** — policy LLM (structured output, retries, taux d'action invalide)
- [x] **T3** — négociation, engagements, réputation publique
- [ ] **T4** — tournoi, OpenSkill, métriques
- [ ] **T5** — boucle de calibration (voir `docs/FINDINGS.md`)
- [ ] **T6** — restitution : replay, graphes, un finding net

## Trois écarts assumés à la spec

1. **La Force est saturante.** `Force = Pop × (base + Guerrier + β·métal + γ·camps)`
   donnerait `Force ∝ cases²`, soit le scaling exponentiel que §12 interdit parce
   qu'il *crée* la stratégie dominante. Les apports technologiques sont donc bornés :
   `β·métal/(métal + m₀)`. Idem pour les points d'intervention, en `log(1 + pop)`
   avec plafond.

2. **La trahison se juge sur l'intention engagée**, pas sur l'issue. §6 interdit au
   dieu d'ordonner une attaque, mais §7 définit l'alliance de guerre par le résultat :
   un allié loyal qui rate son jet de seuil serait compté traître. Le moteur vérifie
   donc l'**action payée** — lisible dans le registre des points d'intervention. Tenir
   parole et perdre la bataille reste tenir parole.

3. **La grille fait 11 cases de côté**, sous la fourchette `[12, 16]` de §2. Mesuré :
   à 12 le premier contact tombe au tour 11.5, à 14 il n'a jamais lieu. Or §2 exige
   des contacts précoces pour que guerres et négociations aient lieu en 15 tours.
   L'intention prime sur la fourchette annoncée.
