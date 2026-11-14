---
name: "style-code-julia"
description: "Conventions pour rendre lisible le code Julia de l'estimation structurelle (SMM) : noms explicites qui gardent la notation du papier, docstrings avec dimensions, commentaires par bloc en anglais et en langage économique, sans toucher au calcul. À utiliser pour écrire, commenter, documenter ou renommer du code Julia de ce dépôt."
---

# Conventions de code Julia pour l'estimation structurelle

Ces règles s'appliquent au code Julia de la racine du dépôt (`*.jl`, `test/*.jl`). `old_version/`
est figé et n'est jamais modifié. Elles s'ajoutent aux règles de `CLAUDE.md` et ne les
remplacent pas : identifiants en lettres latines uniquement, notation χ pour les parts
agrégées, mot « aggregates » et non « head ».

L'objectif est qu'un économiste qui connaît le papier puisse lire une fonction et
retrouver l'équation qu'elle calcule, sans avoir à reconstituer le rôle de chaque variable.

## 1. Le calcul n'est pas touché

L'efficacité du code passe avant la lisibilité. Un travail de lisibilité ne modifie que
les **noms**, les **commentaires** et les **docstrings**.

- Ne pas réécrire les boucles en broadcasting, ni l'inverse. Ne pas retirer ou ajouter de
  `@inbounds`, de préallocation, de conversion de type ou d'annotation de type.
- Ne pas changer l'ordre des opérations, même s'il est mathématiquement équivalent :
  les arrondis changent, et la vérification bit à bit (§5) ne passe plus.
- Ne pas ajouter d'abstraction (struct, fonction auxiliaire, macro) pour la seule lisibilité.
- Si un passage reste illisible sans restructuration, le **signaler** à l'utilisateur au lieu
  de le réécrire.

## 2. Noms

### Objets du modèle : la notation du papier, en lettres latines

Un objet qui a un symbole dans le papier garde ce symbole, translittéré : `alpha`, `kappa`,
`theta`, `chi_L`, `chi_s`, `P_sr`, `P_r`, `c_r`, `T_ls`, `Phi`, `PhiF`, `omega_sr`, `ell_r`,
`pi_r`, `A`. Les indices sont des suffixes, dans l'ordre du papier :

| Suffixe | Sens |
|---|---|
| `_s` | secteur amont |
| `_r` | acheteur aval (index dans `DOWNSTREAM_REGIONS`) |
| `_l` | cellule d'origine (zone d'emploi × secteur) |
| `_a`, `_area` | aire d'attraction |
| `_sr`, `_ls`, `_rs` | produits des précédents, dans l'ordre des indices de la matrice |

La notation n'est valable que si l'objet est défini une fois, dans la docstring de la
fonction qui le crée ou dans l'en-tête du fichier.

### Le reste : des noms explicites

Les variables qui n'ont pas de symbole dans le papier ont un nom descriptif en `snake_case` :
`intermediate_bill`, `model_sector_shares`, `newton_step`, `reference_area`. Les noms opaques
sont à éviter : `tmp`, `res`, `r`, `h`, `xf`, `tg`, `wI`, `Ot`, `gm`, `rv`, `d`.

Exceptions tolérées :
- les indices de boucle qui suivent le tableau ci-dessus (`s`, `l`, `a`, `dr` pour un acheteur
  aval, `g` pour une cellule active), et `it` pour une itération ;
- `tol`, `max_iter`, `damping`, qui sont des conventions universelles des solveurs ;
- les objets mathématiques standard d'un algorithme quand la docstring les définit
  (par exemple le vecteur des inconnues d'un Newton).

### Conventions Julia

- `snake_case` pour les fonctions, variables, arguments et champs de `NamedTuple` ;
  `UPPER_CASE` pour les `const` globales ; `CamelCase` pour les types.
- Un `!` à la fin du nom d'une fonction qui modifie ses arguments.
- `pi` désigne la constante π en Julia : ne pas l'utiliser comme nom de variable
  (écrire `pi_r` ou `sales_shares`).

### Garde-fous d'un renommage

Le code est appelé d'un fichier à l'autre et ses sorties sont lues par Python.
- **Ne jamais renommer une chaîne écrite sur disque** : labels (`PARAM_LABELS`,
  `MOMENT_LABELS`), clés `npz`, noms de fichiers et de dossiers. C'est le contrat avec
  `utils.py`.
- **Ne jamais toucher aux chemins** : `include(...)`, `joinpath(...)`, noms de fichiers.
- Les **arguments mot-clé** et les **champs de `NamedTuple` renvoyés** sont une interface :
  renommer tous les appels et toutes les lectures (`res.champ`) dans le même commit,
  y compris dans `test/*.jl`. Chercher avec `grep -nw` sur toute la racine avant et après.
- Attention au dispatch multiple : une même fonction peut avoir plusieurs méthodes dans
  des fichiers différents (par exemple `profile_targets` dans `profiling.jl` et dans
  `sensitivity.jl`).
- Les noms de fonctions déjà explicites ne sont pas renommés : ils sont cités dans
  `CLAUDE.md`, le README et la documentation.

## 3. Docstrings

Chaque fonction a une docstring `"""` qui contient :
1. la signature, avec ce qui est renvoyé ;
2. ce que la fonction calcule **dans le modèle**, avec l'équation en Unicode
   (`P_sr = Γ_s Φ_sr^{−1/θ}`) ;
3. pour chaque matrice ou vecteur en argument ou en sortie, ses **dimensions et l'ordre
   des indices** (`P_sr (S × R_d)`, `γ (T_COL_DIM × S)`) ;
4. les normalisations (`A_1 = 1`, `T[s, ref] = 1`, parts sur le simplexe) et les cas
   particuliers (colonne de cible nulle, secteur sans importation).

## 4. Commentaires

- Écrits **en anglais**, comme le reste du code et le papier.
- **Un commentaire par bloc logique**, numéroté `# 1)`, `# 2)`… dans une fonction longue.
- Le commentaire dit **pourquoi l'étape existe sur le plan économique**, pas ce que fait la
  ligne. Par exemple : « Sales are pinned at their observed shares, so downstream
  productivity drops out of the spending weights and is solved once, at the end. » plutôt
  que « compute spend_w ».
- Des commentaires en fin de ligne seulement quand une ligne construit un objet économique
  dont le sens n'est pas évident.
- Des phrases complètes et naturelles, sans style télégraphique. Les sigles standard de la
  littérature restent acceptés : CES, EK, SMM, PSO, GE.

## 5. Vérification avant de livrer

1. `git diff` ne modifie aucune ligne de calcul, aucune chaîne écrite sur disque et aucun
   chemin : uniquement des identifiants, des commentaires et des docstrings.
2. `grep -nw` sur chaque ancien nom renommé ne renvoie plus rien dans `*.jl` et `test/*.jl`
   (hors `old_version/`).
3. Il ne reste aucun nom opaque hors des exceptions du §2, et chaque matrice documentée
   a ses dimensions.
4. Si Julia est disponible : les gates `test/test_profiling.jl`, `test/test_foreign.jl` et
   `test/test_sensitivity.jl` passent, et les sorties sont bit-identiques sur l'arbre
   synthétique (`test/make_synthetic_baseline.py`). Sinon, écrire « Not run » dans le
   changelog.
5. Une ligne est ajoutée au changelog de `CLAUDE.md`.
