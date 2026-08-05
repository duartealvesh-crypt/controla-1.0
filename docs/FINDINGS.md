# Constats d'équilibrage

Journal des mesures faites en cours de build. Chaque entrée est un **fait mesuré**,
pas une intuition — §13 interdit explicitement de calibrer à l'intuition.

Ce document est l'ordre de mission de la tranche 5 (sweep).

> **Correction, tranche 4.** Tous les taux de **domination** publiés avant la
> tranche 4 sont faux : le banc de mesure ne faisait pas tourner les positions.
> Les chiffres corrigés sont au constat n° 11, et la domination réelle est de
> **65 %**, non de 33 %. Les autres mesures (rangs, populations, cases, batailles,
> parole) ne sont pas affectées.

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

> Les taux de domination de cette section sont **invalidés** par le constat n° 11.
> Les rangs, populations et cases restent valables.

---

## T4 — le banc de mesure, et ce qu'il a immédiatement trouvé

Trois campagnes de 48 parties (12 seeds × 4 rotations de sièges), archétypes
scriptés, dans `runs/t4-reference`, `runs/t4-military` et `runs/t4-floor`.

### 11. Toutes les mesures de domination antérieures étaient faussées par la position

Le banc ad hoc utilisé pendant l'équilibrage tirait ses permutations de sièges avec
`itertools.permutations(range(4))[:3]`, soit `(0,1,2,3)`, `(0,1,3,2)`, `(0,2,1,3)`.
Ces trois-là **laissent le premier archétype dans le coin 0 à chaque fois**, et le
deuxième dans le coin 1 deux fois sur trois. La mesure comparait donc des
archétypes *et* des positions.

Même panel, mêmes seeds, seule la façon de tourner les sièges change :

| Plan de mesure | Parties | Domination | Répartition |
|---|---|---|---|
| seeds 0-7 × 3 permutations lexicographiques | 24 | **33 %** | 6 / 3 / 8 / 7 |
| seeds 0-7 × 3 rotations | 24 | **54 %** | 3 / 3 / 5 / 13 |
| seeds 1-8 × 4 rotations | 32 | **62 %** | 3 / 3 / 6 / 20 |
| seeds 1-12 × 4 rotations | 48 | **65 %** | 3 / 3 / 11 / 31 |

Le « meilleur résultat obtenu, 33 % » était un artefact. Le tournoi fait désormais
jouer à chaque participant **chaque coin exactement une fois par seed**, et c'est
non négociable : sans rotation, le classement mesure en partie la position.

**Conséquence : le critère (b) de §13 n'est pas atteint.** `turtle` domine à 65 %
pour un idéal de 25 %. La voie technologique reste la stratégie forte, et tout
l'équilibrage de la guerre n'a pas suffi.

Classement de référence (`μ − 3σ`, 48 parties) :

| | μ−3σ | μ | σ | Victoires | Rang moyen |
|---|---|---|---|---|---|
| turtle | **29.75** | 43.77 | 4.67 | 31 | 0.50 |
| expansionist | 17.06 | 29.93 | 4.29 | 11 | 1.23 |
| cooperator | 11.14 | 23.88 | 4.25 | 3 | 1.65 |
| betrayer | −13.05 | 1.93 | 5.00 | 3 | 2.62 |

Panel militaire, mêmes conditions : `turtle` 23 victoires, `conqueror` **12** — mais
le `conqueror` finit dernier au classement OpenSkill malgré ses victoires, avec le
même rang moyen que le `cooperator` (1.88) pour trois fois plus de victoires. C'est
un profil **à forte variance** : il gagne ou il s'effondre, et Plackett-Luce le
sanctionne. Une information que le simple décompte de victoires ne donnait pas.

### 12. Le classement sépare le skill, la chance et l'absence

Panel plancher (`turtle` / `cooperator` / `random_legal` / `noop`), 48 parties :

| | μ−3σ | Victoires | focus | engagement |
|---|---|---|---|---|
| turtle | **35.49** | 35 | 0.83 | 4.14 |
| cooperator | 25.13 | 13 | 0.49 | 4.75 |
| random_legal | 1.53 | **0** | 0.29 | 3.49 |
| noop | −18.22 | **0** | — | 0.00 |

C'est le critère (c) de §13, et il est **atteint** : le hasard ne gagne jamais, et
l'absence est loin derrière le hasard. Le levier divin existe donc bien — un dieu
actif bat un dieu absent, ce qui est la condition pour que le benchmark compare des
modèles et non des économies.

La cohérence discrimine aussi comme voulu : 0.83 pour un archétype qui poursuit une
ligne, 0.29 pour un tirage uniforme. Et `noop` n'obtient **aucun** score
d'exactitude au lieu d'un 1.00 flatteur — le piège du modèle muet est bien évité.

### 13. La faction 0 colonise en premier, donc elle finit la plus exposée

Panel militaire, où le `conqueror` est le **seul** attaquant des 633 batailles : le
coin 0 tombe à 74 % des assauts subis contre 46 % pour le coin 1, à volume d'agression
égal (157 contre 155).

Il a fallu trois vérifications pour savoir ce que ce chiffre dit vraiment.

**1. La rotation est bien équilibrée** — vérifié dans les données, pas seulement dans
le plan : chaque coin héberge chaque archétype exactement 12 fois sur 48 parties, et
la matrice des assauts par couple (attaquant, défenseur) est plate à ±15 %.

**2. Le premier calcul de significativité était faux.** J'avais annoncé 7 σ en
traitant chaque assaut comme un tirage indépendant. Ils ne le sont pas du tout : un
`conqueror` qui prend l'ascendant enchaîne quinze assauts victorieux dans la même
partie. L'unité d'observation est la **partie**, pas l'assaut.

| Coin | Taux de chute (moyenne des taux par partie) | Écart-type inter-parties |
|---|---|---|
| 0 | 73.8 % ± 4.8 | 28.8 % |
| 1 | 50.3 % ± 6.4 | 38.2 % |
| 2 | 48.4 % ± 5.5 | 33.2 % |
| 3 | 65.0 % ± 6.3 | 37.5 % |

Réel, mais à ~2.9 σ, pas 7.

**3. Le contrôle décisif : des panels homogènes.** Quatre policies identiques, 40
parties chacune — tout écart restant vient alors du moteur seul.

| Panel | Victoires par coin | Rang moyen par coin |
|---|---|---|
| `turtle` × 4 | 9 / 9 / 9 / 13 | 1.62 / 1.50 / 1.38 / 1.50 |
| `conqueror` × 4 | 8 / 13 / 10 / 9 | 1.70 / 1.32 / 1.70 / 1.27 |

Dans le bruit. **Le coin n'est donc pas décisif en soi** — le constat initial était
sur-lu.

### Ce qui reste, et qui est réel

Un petit biais mécanique, mesuré sur 96 parties du panel militaire :

| Coin | Part de cases frontalières (fin de partie) | Distance moyenne au centre |
|---|---|---|
| **0** | **42.9 % ± 1.6** | **5.91 ± 0.05** |
| 1 | 39.8 % ± 1.8 | 6.06 ± 0.04 |
| 2 | 37.0 % ± 1.5 | 6.05 ± 0.04 |
| 3 | 38.4 % ± 1.6 | 5.98 ± 0.06 |

**Toutes les passes itèrent les factions par identifiant croissant.** La faction 0
colonise donc la première, pousse le plus loin vers le centre disputé, et termine
avec le territoire le plus **avancé** et le plus **frontalier** — 42.9 % de cases
exposées contre 37.0 %.

Six points d'exposition, c'est peu. Mais un agresseur choisit ses cibles : il frappe
là où c'est ouvert, et convertit ce petit écart structurel en un grand écart de
cases perdues. D'où 74 % contre 46 % en présence d'un `conqueror`, et rien du tout
sans lui.

La rotation moyenne ce biais entre participants, elle ne le supprime pas : elle
ajoute de la variance, et elle n'est pas neutre quand un seul participant est en
position de l'exploiter.

**Correctif pour T5**, à un endroit : départager les revendications simultanées de
cases par un tirage keyé `(seed, tour, "expansion")` plutôt que par l'ordre des
identifiants. Même remarque pour l'ordre d'application à la barrière.

### 14. La métrique de parole est diluée par un engagement qu'on tient sans rien faire

Sur le panel de référence, 48 parties :

| Archétype | Parole | Engagements | Tours honorés | Rompus |
|---|---|---|---|---|
| cooperator | **100 %** | 54 | 201 | 0 |
| betrayer | **99 %** | 54 | 201 | 2 |
| random_legal *(panel plancher)* | **97 %** | 39 | 144 | 4 |

Un tirage uniforme obtient 97 % de parole. La métrique ne discrimine plus rien.

La cause est structurelle : l'alliance **économique** est honorée par défaut, chaque
tour, tant que personne ne pose d'embargo. Ne rien faire, c'est la tenir. Elle
rapporte donc ~4 crédits de parole par engagement, et ces crédits gratuits noient
les rares manquements sur les alliances de **guerre**, seules à exiger une action
payée. C'est exactement l'inverse de ce que §7 veut mesurer.

À noter aussi : 1 602 promesses envoyées pour 54 engagements formés — les
archétypes saturent le canal de négociation, et une promesse coûte trop peu.

Trois pistes pour T5 :
1. Ne créditer une alliance économique qu'**à son échéance**, une fois, plutôt qu'à
   chaque tour — une promesse tenue vaut alors une promesse, pas quatre.
2. Pondérer les deux types : tenir un pacte de guerre demande une dépense, tenir un
   pacte économique demande une abstention.
3. Faire coûter la promesse (un point d'intervention par pledge), pour que
   s'engager soit un arbitrage et non un réflexe.

Le panel diplomatique de T3 discriminait, lui (`coalition_builder` à 44 %), parce
qu'il promet des guerres sur une cible mouvante. Le défaut ne s'était donc pas vu.

### 15. Pourquoi le `conqueror` perd : la décomposition complète

48 parties du panel militaire. Score moyen et part de chaque axe :

| Archétype | Population | Cases | Palier | Bâtiments | **Total** |
|---|---|---|---|---|---|
| turtle | 112.4 → 28.1 (25 %) | 19.3 → 28.9 (26 %) | 2.9 → 28.5 (26 %) | 8.5 → 25.4 (23 %) | **111.0** |
| expansionist | 141.8 → 35.5 (35 %) | 20.9 → 31.3 (31 %) | 1.1 → 11.5 (11 %) | 7.7 → 23.1 (23 %) | **101.3** |
| cooperator | 123.3 → 30.8 (33 %) | 20.2 → 30.3 (32 %) | 1.2 → 12.5 (13 %) | 6.7 → 20.2 (22 %) | **93.8** |
| **conqueror** | 44.6 → 11.1 (13 %) | **29.0 → 43.5 (49 %)** | 1.7 → 17.1 (19 %) | 5.4 → 16.3 (19 %) | **88.1** |

L'écart au `turtle` est de **22.9 points**, et il se décompose exactement :

| Axe | Écart | |
|---|---|---|
| population | **−17.0** | il finit à 40 % de la population des autres |
| palier de recherche | **−11.4** | 1.7 contre 2.9 |
| bâtiments | **−9.1** | 5.4 contre 8.5 |
| cases | **+14.6** | sa seule victoire, et elle est nette |

#### La guerre consomme 89 % de son budget divin

| Archétype | Actions payées | Points dépensés | Répertoire dominant |
|---|---|---|---|
| cooperator | 3 148 | 5 565 | `blessing` ×1955, `call_to_trade` ×885 |
| expansionist | 3 011 | 5 623 | `blessing` ×2612 |
| turtle | 2 948 | 5 526 | `call_to_reflection` ×2578 |
| **conqueror** | **1 924** | 5 202 | `call_to_war` ×1542 (**89 % des points**) |

`call_to_war` coûte **3 points, l'action la plus chère du jeu** (bénédiction 2,
réflexion 2, commerce 1), et c'est la seule qui ne produit **rien** d'économique.
À budget égal, le `conqueror` joue donc un tiers d'actions en moins que les autres —
1 924 contre ~3 000 — et son retard de population est visible **dès le tour 3**
(24 habitants contre 28 à 31), bien avant que la guerre ne coûte quoi que ce soit.

Rendement comparé, en points d'intervention par point de score gagné :

* `turtle` : 107 pts/partie → 28.5 de palier, soit **3.8 pts par point de score** ;
* `conqueror` : 96 pts/partie de guerre → +13.5 de cases au-dessus du plancher,
  soit **7.1 pts par point de score**. La guerre coûte **deux fois** la recherche.

#### Ce sont ses défaites qui le saignent, plus ses victoires

| | |
|---|---|
| Assauts lancés | 633 (13.2 par partie) |
| Gagnés | 360 — **57 %** |
| Morts à l'attaque | 4 431, soit **7.0 par assaut** |
| Population finale | 44.6 |

Le correctif d'attrition du vainqueur a bien pris : écraser ne coûte presque rien.
Mais **43 % des assauts sont des défaites**, et une défaite coûte 29 à 43 % des
engagés. Le `conqueror` tue donc deux fois sa propre population finale au cours
d'une partie, et l'essentiel vient des batailles perdues. Spirale : moins de monde
→ Force plus faible → plus de défaites → moins de monde.

#### Prendre une case détruit son bâtiment

`resolve_battle` pose `tile.building = None` à l'annexion. Sur 360 conquêtes et une
densité de bâtiments d'environ un tiers des cases, ce sont **~120 bâtiments
détruits** sur 48 parties — ~2.5 par partie, soit **7.4 points de score qui
s'évaporent** au lieu de changer de mains. Or l'axe bâtiments est justement celui où
le `conqueror` est en retard (5.4 contre 8.5).

#### Les quatre leviers, par effet attendu

1. **Baisser le prix de `call_to_war` de 3 à 2.** Une constante, et personne d'autre
   dans le panel ne paie cette action. Rend ~30 % d'actions au `conqueror`, ce qui
   attaque directement le trou de population *et* son retard de départ au tour 3.
2. **Le bâtiment survit à la prise.** Convertit la conquête en points sur l'axe
   exact où il manque. Vaut ~7.4 points par partie, soit un tiers de l'écart.
3. **Baisser `attrition_loser` de 0.28 à ~0.20.** Cible ses 273 défaites, qui sont
   la source principale de sa saignée. Le seul archétype qui perd des assauts en
   tant qu'attaquant, c'est lui.
4. **Baisser `score.weights.population` de 0.25 à 0.15** — efficace (+6.8 net pour
   lui) mais c'est compenser dans la mesure au lieu de corriger la mécanique, le
   même argument qui avait fait écarter `case 4.5 / pop 0.12 / rech 7`. Et les cases
   pèsent déjà 49 % de son score.

Les leviers 1 et 2 sont des corrections de mécanique et ne touchent aucun autre
archétype ; le 3 rouvre l'arbitrage déjà mesuré entre volume et viabilité de la
guerre (constat « étape 3 »).

### 16. Les leviers 1 et 2 mis à l'épreuve : un rejeté, l'autre retenu à 0.35

Les deux leviers isolés puis combinés, panel militaire, 12 seeds × 4 rotations :

| Variante | Domination | `conqueror` | Population | Assauts | Gagnés |
|---|---|---|---|---|---|
| référence (3 pts, ouvrage rasé) | 50 % | rang 1.85, 11 v | 49.2 | 623 | 60 % |
| **A** — guerre à 2 points | **56 %** | rang **2.38**, 7 v | **19.6** | 940 | **36 %** |
| **B** — ouvrage capturé | 54 % | rang **1.10**, 26 v | 67.4 | 644 | 64 % |
| A + B | 42 % | rang 1.60, 18 v | 32.0 | 950 | 43 % |

#### A est rejeté : le prix de la guerre est un bouton de VOLUME

Le raisonnement était qu'à 3 points — l'action la plus chère du jeu et la seule sans
contrepartie économique — l'Appel à la guerre étranglait l'économie du belliqueux,
qui y engloutissait 89 % de son budget.

Rendre ces points **n'a pas acheté de la croissance, il a acheté plus de guerre.**
Les actions passent de 1 924 à 2 685 et la totalité du surplus part en appels : 623
assauts deviennent 940, le taux de victoire s'effondre de 60 % à 36 %, la population
finale tombe de 49 à 20. La spirale s'accélère au lieu de se briser.

**Ce prix pilote le volume d'agression, pas la trésorerie du dieu.** C'est un
résultat général sur les archétypes scriptés : leur fonction de score ne réalloue
pas un budget rendu, elle en rachète la même chose. Prix maintenu à 3.

#### B est retenu, mais à 0.35 — à 1.0 il crée le déséquilibre inverse

Balayage du nouveau bouton `combat.building_capture_chance`, prix maintenu à 3 :

| chance | 0.00 | **0.35** | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| **Domination** | 50 % | **40 %** | 44 % | 52 % | 54 % |
| `turtle` | 24 v | 19 v | 18 v | 15 v | 15 v |
| `conqueror` | 11 v | 18 v | 21 v | 25 v | 26 v |
| rang du `conqueror` | 1.85 | 1.48 | 1.35 | 1.23 | 1.10 |

Le bouton fait basculer la domination du technologue au conquérant, et le
croisement est net. **À 1.0, la conquête devient à son tour la stratégie
dominante** (26 victoires sur 48) : on aurait remplacé un déséquilibre par un
autre. Deux tiers des ouvrages brûlent donc encore.

**Retenu : `call_to_war: 3` (inchangé), `building_capture_chance: 0.35`.**

| Panel | Avant | Après |
|---|---|---|
| militaire | 50 % — `conqueror` **dernier** (rang 1.85) | **40 %** — `conqueror` **2ᵉ** (rang 1.48) |
| référence | 65 % | 62 % |
| plancher | 73 % | 73 % (personne n'y conquiert) |

40 % est le meilleur taux de domination mesuré depuis le début sur un banc à
rotations, pour un idéal de 25 %. Le panel de référence bouge à peine : le
`betrayer` conquiert trop peu pour que la règle le concerne.

Reste ouvert : le `cooperator` ne gagne plus que 2 parties sur 48 sur le panel
militaire. Le plafond est mieux partagé, le plancher pas encore.

### 17. Les archétypes étaient aveugles au signe de deux effets

Trouvé en rédigeant le récapitulatif du panel, puis confirmé par un test.

Un archétype pondère des **types d'effet**, jamais des noms d'action — c'est ce qui
rend le panel moddable. Le piège de ce design est qu'un effet peut changer de *sens*
selon son signe. `_effect_key` le corrigeait pour `trade_relation` (commerce ≠
embargo), et le docstring expliquait pourquoi. Deux cas avaient été manqués.

**La bénédiction et la malédiction portaient la même clé.** Toutes deux sont un
`tile_modifier` : +0.70 sur sa propre case, −0.60 sur celle d'un rival. Le
`cooperator` — déclaré `war_desire −0.9`, `embargo −0.9` — n'avait donc aucun moyen
de dire qu'il ne maudissait pas : **219 malédictions sur 48 parties**, 12 % de son
budget divin. L'`expansionist` n'y échappait que par accident, la bénédiction
portant en plus un `trait_call(worker)` qu'il pondère assez haut pour que le softmax
ne tire jamais l'autre.

**Un Appel montait un trait et en baissait un autre, seul le versant positif était
lu.** L'Appel aux armes donne +2 Guerrier pour −1 Résilience ; le `turtle`, qui tient
à sa Résilience, ne voyait pas ce qu'il bradait, et le `conqueror` ne voyait pas
qu'un Appel à la modération lui coûtait son trait Guerrier. Perdre ce à quoi l'on
tient décourage désormais autant que le gagner attire.

Valeurs initiales de la nouvelle clé `curse` : **la position déclarée sur
l'embargo**, puisque les deux consistent à nuire sans faire la guerre. Ce n'est pas
sept nombres inventés, c'est une stance déjà écrite qu'on recopie.

#### Effet mesuré

| | Avant | Après |
|---|---|---|
| Malédictions du `cooperator` | 219 | **0** |
| Ses bénédictions | 1 955 | 2 237 |
| Son score / sa population | 93.7 / 121.8 | 94.7 / 124.5 |
| Ses victoires (panel militaire) | 2 | **2** |
| Domination, panel militaire | 40 % | 40 % |
| Domination, panel de référence | 62 % | **58 %** |

**La prédiction était trop optimiste.** J'annonçais « 657 points d'intervention à
réinvestir » ; ils sont bien réinvestis (+282 bénédictions, +81 appels au commerce)
mais ne rapportent qu'un point de score. Les malédictions n'étaient donc pas si
coûteuses : ce qu'elles gaspillaient, elles le reprenaient en partie aux rivaux. La
correction vaut pour la **cohérence** du panel — un pacifiste qui maudit n'est pas
un pacifiste — plus que pour l'équilibre.

Le correctif du signe des traits n'a, lui, rien changé de mesurable : le `turtle`
jouait déjà 0 Appel aux armes et le `conqueror` 0 Appel à la modération. Le défaut
était latent, pas actif.

#### Effet de bord révélateur : la malédiction est une action dominée

Le `betrayer`, à qui la nouvelle clé donne enfin un appétit cohérent pour le
sabotage (`curse: 1.2`), en lance **502 sur 48 parties** — et tombe de 4 victoires à
**0**, ordinal −8.87 → −23.66.

Maintenant qu'un archétype la veut vraiment, on voit qu'elle ne vaut rien :

| | Bénédiction | Malédiction |
|---|---|---|
| Coût | 2 points | **3 points** |
| Amplitude | +0.70 | **−0.60** |
| Durée | 3 tours | 3 tours |
| Bénéficiaire | soi, à coup sûr | **un rival sur trois** |

Plus chère, plus faible, et son bénéfice est dilué entre trois adversaires dont un
seul est frappé. À traiter en T5 : aligner le coût sur celui de la bénédiction, ou
renforcer `miracles.disaster_malus`.

---

## T5 — la calibration, six axes balayés sur 2 016 parties

`cli sweep` joue un plan de variantes en mode **un levier à la fois**, panel scripté,
zéro appel LLM. Trois passages successifs, chacun décidé par le précédent.

### 18. Un seul levier sur six déplace l'équilibre

Premier passage : 6 axes, 12 variantes, 2 panels, **1 152 parties en 8 min 45**.

| Axe balayé | Domination obtenue | Verdict |
|---|---|---|
| `research.slowdown_knee` 6 / 16 | 44 % / 38 % | ±2 pts — inerte |
| `score.weights.research` 7 | 40 % | inerte |
| `combat.attrition_loser` 0.20 / 0.36 | 40 % / 44 % | inerte |
| `combat.building_capture_chance` 0.20 / 0.50 | 35 % / 44 % | ±5 pts |
| `emergence.war_threshold` 1.2 / 2.0 | 42 % / 42 % | inerte |
| **`emergence.expansion_food_cost` 0.8 / 1.8** | 48 % / **29 %** | **−11 pts** |

Référence : 40 %. Cinq axes sur six ne bougent la domination que du bruit ; le sixième
la déplace de onze points. Sans le sweep, j'aurais continué à tourner les boutons du
combat — c'est là que j'avais passé tout l'équilibrage précédent, et ils ne servent à
rien pour ce critère.

### 19. Le vrai indicateur est le PLANCHER, pas la domination

| `expansion_food_cost` | 0.8 | 1.2 | **1.8** | 2.4 | 3.0 |
|---|---|---|---|---|---|
| Domination (meilleur) | 48 % | 40 % | **29 %** | 38 % | 35 % |
| **Plancher (pire)** | 4 % | 4 % | **21 %** | 19 % | 21 % |
| Étalement | 0.74 | 0.87 | **0.99** | 0.97 | 0.98 |

La domination plafonne dès 1.8 et le reste est du bruit. Ce qui change vraiment, c'est
le plancher : **le quatrième archétype passe de 2 victoires sur 48 à 10.** La
répartition devient 29 / 27 / 23 / 21 % pour un idéal de 25 % chacun.

C'est exactement la distinction relevée plus haut : 40/38/19/4 et 40/20/20/20 ont la
même domination et ne sont pas la même arène. `Criteria` émet donc les trois nombres —
domination, plancher, étalement — et non le seul maximum.

**Retenu : `expansion_food_cost: 1.2 → 1.8`.**

Le mécanisme est direct : un technologue qui met tout dans la Recherche ne peut plus
s'offrir de terres.

| à 1.8 | Cases | Population | Palier | Score |
|---|---|---|---|---|
| cooperator | 20.3 → 17.2 | 124.5 → 127.5 | 1.31 → 1.35 | 94.7 → 92.6 |
| conqueror | 29.8 → 24.6 | 55.5 → 53.0 | 1.98 → 1.92 | 102.5 → 92.7 |
| expansionist | 20.8 → 17.9 | 139.2 → 141.5 | 1.15 → 1.27 | 99.9 → 98.3 |
| **turtle** | 19.1 → **13.9** | 108.9 → 94.4 | 2.75 → 2.75 | 108.3 → 97.5 |

L'écart de score entre le premier et le dernier tombe de 13.6 à 5.7 points.

### 20. La taille de grille ne sert plus à rien — et une hypothèse de T1 est périmée

Deuxième et troisième passages : `expansion_food_cost` 1.8 → 3.0, puis `grid_size`
9 / 10 / 12 à rareté fixée.

| Configuration | Domination militaire | Domination référence | Carte occupée |
|---|---|---|---|
| grille 11, coût 1.2 | 40 % | 58 % | 74 % |
| **grille 11, coût 1.8** | **29 %** | 54 % | 61 % |
| grille 10, coût 1.8 | 31 % | 50 % | 73 % |
| grille 12, coût 1.8 | 29 % | 44 % | 52 % |

Les écarts de grille valent 2 à 3 victoires sur 48 : du bruit. **Grille inchangée
à 11** — la changer serait calibrer sur du bruit, ce que §13 interdit.

En revanche une inquiétude s'est révélée infondée, et une hypothèse de T1 est morte :

| | Batailles/partie | 1ʳᵉ bataille | Parties avec guerre |
|---|---|---|---|
| grille 11, coût 1.2 | 13.3 | **T4.4** | 48/48 |
| grille 11, coût 1.8 | 12.4 | **T4.4** | 48/48 |
| grille 12, coût 1.8 | 12.7 | **T4.4** | 48/48 |
| grille 10, coût 1.8 | 12.7 | **T4.4** | 48/48 |

Le volume de guerre est **totalement insensible** à la rareté comme à la taille de
carte. Le constat n° 1 de T1 — « à 14, les factions ne se touchent jamais, la guerre
est mécaniquement impossible » — n'est plus vrai : depuis que l'adjacence n'est plus
requise pour attaquer, **le contact ne conditionne plus la guerre**. La première
bataille tombe au tour 4.4 quelle que soit la grille.

`grid_size: 11` reste donc un choix, pas une contrainte. La fourchette [12, 16] de §2
redevient jouable si l'on veut une carte plus vaste.

### 21. Les trois critères de §13 sont atteints — chacun sur le panel qui les mesure

Config retenue, 48 parties par panel :

| Panel | (a) horizon | (b) équilibre | (c) skill |
|---|---|---|---|
| **militaire** `cooperator/conqueror/expansionist/turtle` | OK | **OK** — 29 %, plancher 21 %, étalement 0.99 | — |
| référence `cooperator/betrayer/expansionist/turtle` | OK | NON — 54 % | — |
| plancher `turtle/conqueror/random_legal/noop` | OK | NON — 50 % | **OK** — 24.6 d'écart au hasard |

Le panel militaire :

| | μ−3σ | Victoires | Rang |
|---|---|---|---|
| expansionist | 18.37 | 11 | 1.23 |
| turtle | 15.99 | 14 | 1.33 |
| cooperator | 13.34 | 10 | 1.67 |
| conqueror | 1.82 | 13 | 1.77 |

**Le critère (b) ne doit pas être lu sur les deux autres panels, et ce n'est pas une
excuse mais une propriété de leur composition.** Le panel plancher contient `noop` et
`random_legal`, qui ne gagnent jamais par construction : la domination y est
mécaniquement ≥ 50 %. Le panel de référence contient `betrayer`, un **témoin** —
il embargote, rompt ses pactes et paie un coût social par conception, pour que la
trahison soit mesurable. Il finit à 0 victoire sur 48 et à −21.6 d'ordinal.

Chaque panel a une fonction : le **militaire** mesure l'équilibre (quatre voies
viables), le **référence** mesure la parole (§7), le **plancher** mesure le levier
divin (skill contre hasard). Les évaluer tous les trois sur les trois critères
produit deux faux échecs.

### Reste ouvert pour T6 et au-delà

* La métrique de parole ne discrimine plus (constat n° 14) : l'alliance économique
  est honorée par inaction, et un tirage uniforme obtient 97 %.
* La malédiction est une action dominée (constat n° 17).
* La faction 0 finit la plus exposée, les passes itérant par identifiant (constat n° 13).
* Aucune élimination en 144 parties : la voie « éliminer un rival » est inerte à
  15 tours.

### 18. Punir le leader est jouable, efficace — et perdant

Question posée avant de mesurer : maintenant qu'il existe deux trajectoires
gagnantes, un dieu peut-il être amené à agresser celui qui gagne ? Trois conditions,
mesurées séparément.

#### a. Le vainqueur est-il identifiable ? Partiellement, et c'est voulu

Ce qu'un dieu voit d'un rival : `tiles`, `buildings` (types compris),
`at_war_with_me`, `trade_relation`, `reputation`. Ce qu'il **ne voit pas** :
population, recherche, palier, Force. Or la moitié du score se joue sur les deux
axes invisibles (population 25 %, palier 26 %).

| Estimateur du leader | Panel militaire | Panel de référence |
|---|---|---|
| le plus de cases | **31 %** | 48 % |
| cases + bâtiments | **73 %** | 79 % |

Juger au territoire désigne le mauvais joueur deux fois sur trois. L'indice utile
est la **composition** des bâtiments, qui est publique :

| | camp | cité | champ | **labo** |
|---|---|---|---|---|
| cooperator | 0.98 | 2.56 | 2.23 | 0.92 |
| **conqueror** | **2.08** | 2.56 | 2.29 | 1.10 |
| expansionist | 0.92 | 2.94 | 2.83 | 0.83 |
| **turtle** | 1.06 | 2.60 | 2.60 | **2.02** |

Le technologue a deux fois plus de laboratoires que quiconque, le conquérant deux
fois plus de camps. L'inférence est disponible, mais elle demande de lire une
composition et non une taille.

#### b. Est-il punissable ? Oui, sans investissement préalable

Force en fin de partie, 48 parties :

| | Force | Force engagée | trait Guerrier |
|---|---|---|---|
| cooperator | 129.3 | 69.5 | 0.53 |
| conqueror | **106.0** | **82.6** | 1.28 |
| expansionist | 137.0 | 72.8 | 0.51 |
| turtle | 114.9 | 62.5 | 0.55 |

Les Forces sont **quasi égales** : la population domine la formule et les apports
technologiques saturent (écart assumé n° 1). Le conquérant a même la Force totale la
plus faible — sa population s'est effondrée — et ne devance les autres qu'en *ardeur*.

Conséquence : deux dieux pacifiques alignent ~140 de Force engagée contre 82.6, soit
un ratio de 1.69 qui donne, au contest de Tullock à `m = 2.5`, **79 % de victoire**.
Punir ne demande pas quinze tours de préparation militaire, seulement de décider
d'employer la population qu'on a déjà.

Le délai, en revanche, est serré : franchir `war_threshold` demande un ou deux tours
d'Appels, la marche un ou deux de plus. Décider au tour 10, c'est frapper au tour 13
— une ou deux frappes coordonnées par partie, pas davantage.

#### c. Est-ce que ça paie ? Non — et c'est LE problème

Deux panels, 48 parties chacun, les deux trajectoires gagnantes contre deux
comportements passifs puis contre deux comportements réactifs (`coalition_builder`
vise le plus gros territoire, `tit_for_tat` rend les coups).

| | Contrôle (2 passifs) | Coalition (2 réactifs) |
|---|---|---|
| **Domination** | **42 %** | **73 %** |
| `turtle` | 20 v — score 107.6 | **35 v** — score 107.8 |
| `conqueror` | 18 v — score 102.4 | 8 v — score **70.3** |
| les deux autres | 5 v / 5 v — score ~97 | 3 v / 2 v — score **~63** |

Les réactifs ont bien coordonné leur agression : 472 assauts sur le conquérant
contre 336 sur le technologue. Ils ont bien **réussi** — le conquérant tombe de
102.4 à 70.3 et perd 10 victoires. Et ils ont **perdu** : scores 63.0 et 62.3, les
deux plus bas du panel, taux de victoire au combat de 43 et 45 %.

Le technologue, lui, n'a rien fait et passe de 20 à **35 victoires sur 48**.

C'est le problème du **bien public** dans sa forme la plus nue : punir le leader
coûte à celui qui punit et profite à celui qui s'abstient. Un modèle qui raisonne
correctement finira donc par ne jamais frapper — et l'intransitivité que §12 exige
n'existera pas.

Le partage du butin de l'alliance de guerre (§7) est censé compenser ; mesuré, il
n'y suffit pas. Pistes pour T5 :
1. **Augmenter la prise sur le leader** — butin, recherche volée, ou ouvrages
   capturés qui montent avec l'écart de score du vaincu.
2. **Faire payer l'abstention** : une guerre qui dure devrait coûter à tous
   (interruption du commerce, tension), pour que rester en dehors ne soit pas gratuit.
3. **Rendre le palier de recherche visible** — l'invisibilité protège aujourd'hui
   exactement la stratégie qui n'a pas besoin de protection.
