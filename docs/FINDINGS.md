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
