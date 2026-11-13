"""Laboratoire d'équilibrage : analyse, optimisation, RL.

Règle de séparation, vérifiée par un test (`lab/tests/test_boundary.py`) :

* le laboratoire n'importe du moteur **que** `controla.api` (jouer une partie,
  lire les paramètres d'un archétype) ;
* le moteur n'importe **jamais** le laboratoire.

On peut donc supprimer ce dossier sans que le jeu s'en aperçoive.
"""
