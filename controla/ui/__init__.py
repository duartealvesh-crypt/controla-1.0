"""Interface web : visionneuse de replay, tableau de campagne, partie jouable.

Trois usages d'un même moteur, servis par une seule application locale :

* **rejouer** une partie stockée, tour par tour, carte comprise ;
* **lire** une campagne — classement OpenSkill, les cinq métriques §12, les
  trois critères §13 ;
* **jouer** un siège soi-même contre le panel scripté ou contre des modèles.

Rien de cette couche ne remonte dans le moteur : elle lit `runs/` et pilote un
`ControlaEnv` comme n'importe quelle autre policy. L'arène ignore qu'un
navigateur la regarde.
"""
