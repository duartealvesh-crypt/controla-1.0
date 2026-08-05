# Constats d'équilibrage

Journal des mesures faites en cours de build. Chaque entrée est un **fait mesuré**,
pas une intuition — §13 interdit explicitement de calibrer à l'intuition.

Ce document est l'ordre de mission de la tranche 5 (sweep).

---

## T1 — mesures sur 24 parties scriptées (8 seeds × 3 permutations)

Panel : `cooperator`, `betrayer`, `expansionist`, `turtle`.

### 1. Taille de grille — corrigé pendant T1

| Grille | Premier contact | Guerres | Batailles |
|--------|-----------------|---------|-----------|
| 10 | T7.7 | 42 | 33 |
| 11 | T7.8 | 32 | 24 |
| 12 | T11.5 | 15 | 12 |
| 14 | jamais | 0 | 0 |

La spec §2 suggérait `GRID_SIZE ∈ [12, 16]` tout en exigeant des contacts **précoces**.
Les deux sont incompatibles à 15 tours avec une expansion d'une case par tour : à 14,
les factions ne se touchent jamais et la guerre est mécaniquement impossible.

**Retenu : `grid_size: 11`.** L'intention de §2 prime sur la fourchette annoncée.

### 2. `turtle` est une stratégie dominante — À TRAITER EN T5

`turtle` gagne **24 parties sur 24**. C'est précisément ce que §12 déclare fatal :
« si un build écrase tout, le benchmark est mort ».

Décomposition du score moyen par axe :

| Archétype | Population | Cases | Recherche | Bâtiments | Total |
|---|---|---|---|---|---|
| cooperator | 33.7 | 35.0 | 12.3 | 22.9 | 103.8 |
| betrayer | 12.9 | 37.4 | 10.0 | 15.1 | 75.5 |
| expansionist | 34.4 | 34.6 | 11.3 | 24.4 | 104.7 |
| **turtle** | 33.8 | 35.4 | **60.2** | 27.1 | **156.5** |

**La recherche est tout l'écart.** Sur les trois autres axes les archétypes sont à
égalité ; turtle prend 5× plus de recherche que tout le monde.

Cause structurelle : **la recherche est le seul axe non plafonné.** La population est
bornée par `K`, le territoire est borné par le nombre de tours, les bâtiments par le
bois — la recherche, elle, s'accumule linéairement sans rendement décroissant. Or §12
exige « rendements décroissants ou plafonnés — jamais exponentiels ».

Pistes à tester au sweep, dans cet ordre :
1. Rendement décroissant sur le gain de recherche (l'analogue du `_saturating` déjà
   utilisé pour la Force).
2. Scorer les **paliers atteints** plutôt que les points bruts — cela plafonne l'axe
   à 3 et rend la tech comparable aux autres voies.
3. Baisser `research.intelligence_weight` (l'effet d'un Appel à la réflexion compose
   avec lui-même sans limite).

### 3. La guerre est nette-négative — À TRAITER EN T5

`betrayer` finit à **51.5** de population contre ~135 pour les non-belliqueux.
L'attrition du vainqueur mange plus que le butin et l'annexion ne rapportent.

C'est un frein anti-snowball qui a trop mordu : §12 veut que « tech ← rush précoce »,
donc l'agression précoce doit pouvoir **payer** contre un turtle. Actuellement elle ne
paie jamais, ce qui supprime l'intransitivité et laisse la tech sans contre.

Pistes : baisser `combat.attrition_winner`, monter `combat.loot_fraction`, ou valoriser
davantage l'annexion (elle est déjà comptée dans l'axe territoire, mais celui-ci ne
discrimine presque pas — voir ci-dessous).

### 4. L'axe territoire ne discrimine presque rien

Écart-type de 2.10 case sur une moyenne de 23.7. Tout le monde atteint ~24 cases quelle
que soit la taille de la grille : l'expansion est plafonnée par les **15 tours** (une
case par tour depuis 9), pas par l'espace disponible.

Conséquence : 35 points de score identiques pour tous, soit un axe qui pèse lourd mais
n'apporte aucun signal. Les annexions de guerre sont la seule source de variance — d'où
le lien avec le constat 3.

Piste : permettre plusieurs cases par tour quand l'envie dépasse largement le seuil,
pour que le territoire redevienne une vraie voie de victoire contestée.

---

## T2 — mesures sur les appels LLM réels

Modèle testé : `llama-3.3-70b-versatile` (Groq, palier gratuit), 1 siège LLM contre
3 archétypes, 15 tours.

### 5. Le budget de tokens est LA contrainte de la campagne — À TRAITER EN T4

| Mesure | Valeur |
|---|---|
| Prompt système (règlement) | ~2 400 tokens, **identique à chaque appel** |
| Prompt utilisateur (état du tour) | ~520 tokens |
| Appels par partie et par siège LLM | 15 (un par tour) |
| Tokens par partie et par siège | **~45 000** |

Une partie à 4 sièges LLM coûte donc ~180 000 tokens. Le palier gratuit de Groq
plafonne à **100 000 tokens par jour** : une seule partie complète l'épuise. Une
campagne de 100 parties demanderait ~18 millions de tokens par modèle.

Le système représente **82 %** de chaque appel. Trois leviers, par ordre d'efficacité :

1. **Cache de prompt côté fournisseur.** Le message système est construit une fois
   par partie et ne change jamais — il est déjà taillé pour ça. Anthropic et OpenAI
   le facturent ~10× moins cher en lecture de cache ; Groq ne le propose pas.
2. **Paralléliser la ronde.** Les quatre dieux planifient sur des états qui ne
   dépendent pas les uns des autres (résolution simultanée, §1) : leurs appels
   peuvent partir ensemble. Gain en temps mural ×4, pas en tokens.
3. **Baisser `llm.max_legal_listed`.** Réduit le prompt utilisateur, soit 18 % du coût.

### 6. Le modèle testé est passif

Sur 15 tours avec 6 à 9 points d'intervention disponibles chaque tour, le modèle a
renvoyé **10 plans vides** et n'a planifié que **6 coups au total**. Quand il agit,
son raisonnement est cohérent (« augmenter l'Intelligence pour accélérer la
Recherche ») et son exactitude est de **100 %** : aucun index hors masque.

Ce n'est pas un défaut du moteur, c'est une mesure — et exactement le genre de signal
que le benchmark doit produire. À confirmer sur d'autres modèles avant d'en conclure
quoi que ce soit.

### 7. La classification d'erreur a été affinée sur un cas réel

Un premier jet rangeait `BadRequestError` avec les erreurs d'authentification, sous
l'étiquette « ce modèle ne jouera jamais ». Une campagne a produit 1 `BadRequest`
pour 15 appels réussis : le message criait au loup. Les erreurs sont désormais
séparées en `fatal` (clé, nom de modèle, accès — le modèle ne jouera jamais),
`bad_request` (cette requête-ci a déplu) et `transient` (déjà retentée par le SDK).


---

## T3 — la métrique de parole, calibrée sur 20 parties scriptées

### 8. Sans cible nommée, le compteur de trahisons ne discriminait pas

Premier jet : l'alliance de guerre ne nommait pas d'ennemi. Quand un allié appelait
à la guerre et pas l'autre, le moteur devait blâmer **arbitrairement le silencieux**
— qui n'avait pourtant promis de suivre aucune cible en particulier.

Résultat mesuré : traître 87 %, coopérateur 86 %. Le compteur mesurait du bruit.

**Correction** : l'engagement de guerre **nomme sa cible** (§7 dit littéralement
« la MÊME cible »). Les deux dieux promettent alors exactement la même chose, et
celui qui ne l'a pas payée est sans ambiguïté celui qui a manqué à sa parole.

### 9. Sans réciprocité de la part du panel, aucun engagement ne se formait

Un engagement exige deux promesses miroir. Les archétypes ne lisaient pas leur boîte
de réception : **5 engagements sur 20 parties**, trop peu pour mesurer quoi que ce
soit. Répondre à une main tendue est devenu une conduite de base du panel.

### 10. Un pacifiste qui signe une alliance de guerre la trahit par inaction

Le coopérateur accumulait 11 trahisons involontaires : il signait des pactes de
guerre qu'il n'honorerait jamais, puisqu'il n'appelle jamais à la guerre. Choisir le
**type** d'engagement cohérent avec sa conduite fait partie de la conduite — chaque
archétype a désormais une appétence par type.

### Résultat

Sur 20 parties, 99 engagements formés :

| Archétype | Tenues | Trahies | Taux de parole |
|---|---|---|---|
| cooperator | 106 | 0 | **100 %** |
| tit_for_tat | 71 | 3 | 96 % |
| betrayer | 104 | 12 | 90 % |
| coalition_builder | 27 | 34 | **44 %** |

Le `coalition_builder` est le plus déloyal, et c'est cohérent : il promet des guerres
contre « le leader », or le leader change en cours de partie. Promettre sur une cible
mouvante, c'est promettre ce qu'on ne pourra pas tenir — le benchmark le capte.

---

## Refonte de la recherche

Trois changements décidés après le constat n° 2, appliqués ensemble.

### Le gain ralentit avec le cumul

```
gain_effectif = gain_brut / (1 + (recherche / 10.0) ^ 2)
```

Le genou est calé sur le **palier 2** : au-delà de la deuxième amélioration, chaque
point coûte beaucoup plus cher. La recherche cumulée croît alors en racine cubique
du temps — pratiquement bornée sur 15 tours, sans mur artificiel.

| Recherche accumulée | 0 | 4 | 10 | 18 | 25 | 40 |
|---|---|---|---|---|---|---|
| Multiplicateur de gain | ×1.00 | ×0.86 | **×0.50** | ×0.24 | ×0.14 | ×0.06 |

### Les bénéfices sont plus modestes

`tier2_food_bonus` 0.25 → **0.12**, `tier3_consumption_cut` 0.15 → **0.08**.
Un palier doit ouvrir une voie, pas la rendre écrasante. Le palier 1 (le Métal)
n'a pas bougé : c'est la seule route technologique vers la Force, et la voie
militaire est déjà déficitaire (constat n° 3).

### Le score compte les paliers, pas les points

`score_components["research"]` renvoie `research_tier` (0 à 3), avec un poids de
10.0 par palier. L'axe est ainsi **borné comme les autres** : population par `K`,
territoire par les 15 tours, bâtiments par le bois.

### Effet mesuré — 24 parties, même panel et mêmes seeds

| | Avant | Après |
|---|---|---|
| Victoires de `turtle` | **24 / 24** | **11 / 24** |
| Écart de score au 2ᵉ | 52 pts | 14 pts |
| Écart sur l'axe recherche | 48 pts | 17 pts |
| Recherche brute de `turtle` | 60.2 | 3.0 paliers (~21 pts bruts) |

Répartition des victoires : turtle 11, cooperator 6, expansionist 6, betrayer 1.
La voie tech reste la plus forte — ce qui est sain pour une spécialisation — mais
elle n'écrase plus. La stratégie dominante que §12 déclare fatale a disparu.

**Reste ouvert pour T5** : la voie militaire est toujours déficitaire (`betrayer`
finit à 63 de population contre ~150, une seule victoire), et l'axe territoire ne
discrimine toujours pas (23.5 à 23.8 cases pour tout le monde).

---

## Refonte de la guerre, étape 1 : les pertes portent sur la force engagée

`engaged_population(cfg, faction) = population × ρ(Guerrier)` est désormais la base
de l'attrition, à la place de la population totale. Seuls les habitants partis au
combat peuvent tomber — et l'ardeur devient un arbitrage réel : engager 85 % de son
peuple, c'est risquer 85 % de son peuple.

| | Avant | Après |
|---|---|---|
| Victoires de l'attaquant | 32 % | **42 %** |
| Pertes de l'attaquant vainqueur | 17.3 | **7.6** |
| Population du `betrayer` | 63 | **85** |
| Rang moyen du `betrayer` | 2.67 | 2.88 |

Le levier est réel mais il ne suffit pas : le `betrayer` reste dernier.

### Le vrai blocage : le volume de conquête est négligeable

Aucun bouton de config ne déplace le rang du `betrayer` — testé sur 24 parties
chacun :

| Variante | Victoires attaquant | Rang `betrayer` |
|---|---|---|
| référence | 42 % | 2.88 |
| Guerrier hors qualité (`warrior_weight: 0`) | 35 % | 2.92 |
| ardeur aplatie (`engaged_max: 0.50`) | 43 % | 2.88 |
| ardeur relevée (`engaged_min: 0.45`) | 32 % | 2.92 |
| attrition vaincu 0.45 → 0.28 | 41 % | 2.88 |
| butin 0.35 → 0.60 | 42 % | 2.88 |
| conquête valant 3.0 au lieu de 1.5 | 42 % | 2.88 |

La cause est en amont du combat :

| Archétype | Cases finales | Conquises | Perdues | Net |
|---|---|---|---|---|
| cooperator | 23.7 | 0 | 7 | −7 |
| **betrayer** | **25.2** | **32** | 0 | **+32** |
| expansionist | 23.4 | 0 | 13 | −13 |
| turtle | 23.5 | 0 | 12 | −12 |

**1.3 annexion par partie**, contre ~15 cases obtenues **gratuitement** par
l'expansion (une par tour). Le `betrayer` termine avec +1.5 case d'avance, soit
**+2.25 points de score**, pour ~80 habitants perdus, soit **−20 points**.

Monter la valeur d'une case ne peut rien y faire : tout le monde en a ~24, donc le
poids s'applique identiquement à tous. Tant que la conquête rapporte une case là où
l'expansion en donne quinze, la voie militaire ne peut pas payer.

Trois directions possibles, par ordre d'effet attendu :
1. **Contraindre l'expansion libre** pour que le territoire redevienne rare et
   contesté (c'est aussi le constat n° 4 : l'axe ne discrimine pas).
2. **Faire qu'une victoire prenne plus qu'une case** (front entier, ou la case plus
   ses voisines non défendues).
3. **Faire que la conquête rapporte autre chose** que du territoire — le bâtiment
   pris intact plutôt que détruit, ou une part de la Recherche du vaincu.

---

## Refonte de la guerre, étape 2 : rareté du territoire, butin de savoir, portée

Trois changements appliqués ensemble.

1. **Coloniser coûte de la nourriture**, proportionnellement au territoire déjà
   tenu : `coût = 1.2 × nb_cases`. Un empire affamé cesse de s'étendre.
2. **Une victoire prélève une part de la Recherche du vaincu** (`0.40 × marge`).
   La conquête devient une route vers le palier 1, donc vers le Métal. Le vaincu
   perd des travaux mais jamais un palier atteint.
3. **L'adjacence n'est plus requise pour attaquer.** Toute case ennemie est une
   cible ; une case non frontalière coûte un tour de marche de plus, ce qui allonge
   d'autant la fenêtre de réaction du défenseur.

### Effet mesuré — 24 parties

| | Étape 1 | Étape 2 |
|---|---|---|
| Batailles | 76 | **276** (dont 213 assauts lointains) |
| Conquêtes du `betrayer` | 32 | **109** |
| Cases finales des non-conquérants | 23.4 – 23.7 | **21.7 – 22.2** |
| Cases finales du `betrayer` | 25.2 | 25.2 |
| Victoires du `betrayer` | 1 | **4** |
| Rang moyen du `betrayer` | 2.88 | **2.33** |
| Pillages de recherche | — | 107 |
| Palier du `betrayer` / `cooperator` | — | 1.38 / 1.17 |

Le territoire est enfin **disputé** : les pacifiques reculent à ~22 cases pendant
que le conquérant tient 25.2. La conquête nourrit aussi la technologie, comme voulu.

### Deux effets de bord à surveiller

**Le volume de guerre a explosé** : 11.5 batailles par partie contre 3.2 avant. La
guerre est devenue l'état par défaut plutôt qu'une décision. Boutons :
`emergence.war_threshold`, `war_decay`, ou `combat.distant_march_turns`.

**`turtle` s'est renforcé** (10 → 13 victoires sur 24). Tout le monde se bat et
saigne, sauf lui : sa population tient à 125 quand celle du `betrayer` tombe à 49.
La voie militaire reste dominée par la voie technologique — le `betrayer` gagne du
territoire mais le paie en population, et le score final ne bascule pas.

---

## Refonte de la guerre, étape 3 : attrition du vaincu et seuil de guerre

Balayage `attrition_loser` × `war_threshold`, 12 parties par case, puis validation
des meilleurs candidats sur 24 parties.

**Retenu : `attrition_loser: 0.28`, `war_threshold: 1.60`.**

### Ce que chaque bouton pilote

`war_threshold` commande le **volume** de guerre, presque linéairement :

| Seuil | 0.80 | 1.20 | 1.60 | 2.20 |
|---|---|---|---|---|
| Batailles par partie | 10.8 | 6.5 | 2.9 | 0.8 |

`attrition_loser` commande la **viabilité** du guerrier — mais seulement là où la
guerre est fréquente. À seuil 0.80, le rang du `betrayer` passe de 2.17 (attrition
0.45) à 1.67 (attrition 0.20). À seuil 1.60, l'attrition ne change plus rien : le
guerrier n'a plus assez d'occasions pour que son avantage compte.

**Les deux boutons tirent donc en sens opposés.** Baisser l'attrition rend la guerre
payante ; monter le seuil la rend rare, ce qui reprend au guerrier ce qu'on vient
de lui donner.

### Effet sur 24 parties

| | Référence (0.45 / 0.80) | Retenu (0.28 / 1.60) |
|---|---|---|
| Batailles par partie | 11.5 | **3.7** |
| **Domination** (part de victoires du meilleur) | **54 %** | **38 %** |
| Victoires `turtle` | 13 | **7** |
| Victoires `expansionist` | 6 | 9 |
| Victoires `cooperator` | 1 | 7 |
| Victoires `betrayer` | 4 | 1 |
| Population du `betrayer` | 49 | **93** |

L'emprise de la voie technologique est **brisée** : trois archétypes se partagent
désormais les victoires (9 / 7 / 7), là où `turtle` en prenait 13 sur 24. C'est le
critère (b) de §13 — « aucune stratégie ne domine » — qui progresse de 54 % à 38 %,
pour un idéal de 25 %.

### Le `betrayer` reste dernier, et le panel y est pour quelque chose

Sa population a récupéré (49 → 93) : il ne se saigne plus. Mais il ne gagne qu'une
partie sur 24.

Observation à verser au dossier : **le panel n'a pas d'archétype militaire pur.**
`betrayer` mêle agression et traîtrise — il embargote, rompt ses engagements et
paie un coût social par conception. On ne peut donc pas conclure « la voie militaire
n'est pas viable » à partir de lui. Il manque un `conqueror` : belliqueux, loyal,
sans embargo, qui testerait la conquête seule.

---

## L'archétype `conqueror` : le contrôle qui manquait au panel

Belliqueux, **loyal**, et qui ne pose jamais d'embargo. `betrayer` mêlait agression
et traîtrise ; on ne pouvait donc pas savoir si sa dernière place venait de sa
belligérance ou de son coût social. `conqueror` isole la conquête.

### Il conquiert beaucoup, et perd quand même

Panel `cooperator` / `conqueror` / `expansionist` / `turtle`, 24 parties :

| Archétype | Rang | Victoires | Cases | Population |
|---|---|---|---|---|
| turtle | 0.50 | **15** | 20.4 | 116.8 |
| expansionist | 1.33 | 5 | 21.4 | 135.0 |
| cooperator | 1.71 | 3 | 20.4 | 119.6 |
| **conqueror** | **2.46** | 1 | **27.8** | **30.4** |

159 conquêtes sur 24 parties, et un avantage territorial réel de **+7 cases**. Mais
sa population tombe au quart de celle des autres.

Le verdict est enfin **attribuable sans ambiguïté** : la voie militaire échoue par
**attrition**, pas par coût social. Arithmétique : +7 cases × 1.5 = +10.5 points,
contre un déficit de ~90 habitants × 0.25 = −22.5 points.

Effet de bord : le `conqueror` étant bien plus belliqueux que le `betrayer`, le
volume repasse à 11.9 batailles par partie et la domination du `turtle` remonte à
62 %.

### Deux défauts révélés par ce test

**Le compteur de parole mesurait le choix d'allié, pas la parole.** Quand un pacte
était rompu, seul le traître était marqué ; le partenaire qui avait bien payé n'était
jamais crédité. Un archétype de loyauté maximale affichait ainsi **12 %** de parole.
Corrigé : le loyal est désormais crédité d'une promesse tenue. `conqueror` passe à
53 %, `betrayer` à 56 %.

**Être belliqueux ne suffit pas à être un allié fiable.** S'abstenir de frapper un
allié ne tient pas un pacte de guerre : il faut payer l'appel contre **la cible
promise**. Un biais dédié (`_promise_bias`) dirige désormais l'agression vers la
cible nommée, pondéré par la loyauté de l'archétype.

### Le poids du territoire redevient un levier

J'avais mesuré que monter `score.weights.tiles` ne changeait rien — c'était exact
**tant que tout le monde finissait à ~23 cases**. Depuis que coloniser coûte, le
territoire discrimine (27.8 contre 20.4), et le poids agit de nouveau :

| Poids | Domination | conqueror | turtle |
|---|---|---|---|
| case 1.5 / pop 0.25 (actuel) | 62 % | 2.46 | 0.50 |
| **case 3.0 / pop 0.25** | **46 %** | 1.88 | 0.83 |
| case 3.0 / pop 0.12 | 54 % | 1.50 | 0.67 |
| case 4.5 / pop 0.12 | 50 % | **1.25** | 1.00 |
| case 4.5 / pop 0.12 / rech 7 | 50 % | **1.17** | 1.12 |

Deux optima distincts : `case 3.0` minimise la domination (46 %), tandis que
`case 4.5 / pop 0.12 / rech 7` rend la voie militaire réellement compétitive
(rang 1.17, deuxième) au prix d'une domination à 50 %. Décision à prendre.

---

## Refonte de la guerre, étape 4 : les pertes du vainqueur tendent vers zéro

### Le défaut : un plancher là où il fallait une décroissance

La forme précédente, `base × (1 + w·(1 − marge))`, **ne descendait jamais sous
`base`**. Écraser un adversaire coûtait donc 15 à 18 % de ses engagés, exactement
comme une victoire arrachée. Enchaîner les batailles saignait mécaniquement, quel
que soit le rapport de force — la vraie raison pour laquelle la voie militaire
n'était pas viable.

### La forme retenue : `base × (1 − marge) ^ k`

Multiplicative, elle tend vers zéro. La marge étant `|f_A − f_B| / (f_A + f_B)`,
elle est invariante d'échelle : cent contre vingt et cinquante contre dix donnent
la même marge (0.667), donc le même coût.

| Rapport | Marge | Vainqueur | *(avant)* | Vaincu |
|---|---|---|---|---|
| 50 vs 50 | 0.000 | 15.0 % | 23.2 % | 43.4 % |
| 70 vs 30 | 0.400 | 9.0 % | 20.0 % | 37.2 % |
| **100 vs 20** | 0.667 | **5.0 %** | 17.8 % | 33.1 % |
| **50 vs 10** | 0.667 | **5.0 %** | 17.8 % | 33.1 % |
| 200 vs 10 | 0.905 | 1.4 % | 15.8 % | 29.5 % |

Le bruit est passé d'absolu à **multiplicatif** au passage : un sigma de 0.06 en
absolu aurait fait varier un taux de 5 % entre 0 et 11 %, soit plus que la valeur
elle-même.

Le vaincu conserve sa forme additive autour d'un plancher élevé : perdre doit rester
grave, et l'on rompt plus tôt face à un rouleau compresseur qu'au terme d'un
affrontement disputé.

### Effet mesuré — 24 parties, poids de score INCHANGÉS

Panel militaire (`cooperator` / `conqueror` / `expansionist` / `turtle`) :

| | Avant | Après |
|---|---|---|
| **Domination** | 62 % | **38 %** |
| Rang du `conqueror` | 2.46 | **1.67** |
| Victoires du `conqueror` | 1 | **8** |
| Population du `conqueror` | 30.4 | **56.2** |
| Victoires du `turtle` | 15 | **9** |

Panel de référence (`cooperator` / `betrayer` / `expansionist` / `turtle`) :
**domination 33 %**, victoires réparties 6 / 3 / 8 / 7.

La voie militaire est enfin **compétitive** : le `conqueror` gagne 8 parties contre
9 au `turtle`. Et 33 % de domination est le meilleur résultat obtenu — pour un idéal
théorique de 25 % à quatre joueurs.

**Aucun poids de score n'a été touché.** Corriger la mécanique valait mieux que
compenser dans la mesure : les poids `case 4.5 / pop 0.12 / rech 7` deviennent
inutiles, et le score reste multi-axes comme §10 l'exige.
