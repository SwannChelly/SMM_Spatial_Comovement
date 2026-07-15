# Initialisation de T : inverser les avantages comparatifs à partir des parts

Cette note explique **pourquoi**, une fois qu'on se donne un *prior* sur le paramètre de
coût commercial `α`, on peut retrouver **précisément** (et de façon **unique**) le vecteur
des avantages comparatifs `T`, à partir des seules parts d'approvisionnement observées.

Elle est écrite pour un lecteur de niveau prépa : on suppose connus l'algèbre linéaire,
les suites récurrentes, un peu de convexité, mais rien de plus. Le fil directeur est :

1. **poser le problème** — conditionnellement à `α`, existe-t-il un `T` unique ? ;
2. **la solution** — oui, et voici comment le trouver ;
3. **le lien avec le code** — l'initialisation implémentée *est-elle* Sinkhorn ? Sinkhorn
   est-il « le mieux » ? faut-il plutôt attaquer directement la forme convexe ?

Le code concerné est `invert_T_from_gamma` dans `load_parameters.jl` (SECTION 10b).

---

## 1. Le problème

### 1.1 Le décor

On a `N` régions. Chaque région `r'` possède un **avantage comparatif** `T_{r'} > 0`
(sa « productivité fondamentale » à la Eaton–Kortum, EK). Le coût de transport d'une
région vendeuse `r'` vers une région acheteuse `r` est une loi de puissance de la distance :

$$
\tau_{r'r} = d_{r'r}^{\,\alpha}, \qquad d_{r'r} = d_{rr'} \;(\text{symétrique}).
$$

Le paramètre `α` gouverne **à quel point la distance décourage le commerce**. On se donne
`α` (c'est le *prior* du titre) ; on le suppose donc **connu et fixé** dans toute cette note.
On note `θ` le paramètre de dispersion de Fréchet (dans le code `θ = 1`), et on résume la
géographie par le **noyau de coût**

$$
K_{r'r} = \tau_{r'r}^{-\theta} = d_{r'r}^{-\alpha\theta} \;>\; 0 .
$$

Point crucial : **dès que `α` est fixé, `K` est entièrement connu** (il ne dépend que des
distances et de `α`). C'est tout l'intérêt de conditionner sur `α` : le problème d'inversion
devient un problème sur `K` fixe.

### 1.2 Ce que dit le modèle EK

La part des achats de la région `r` qui provient de la région `r'` (part bilatérale) est

$$
\gamma_{r'r} = \frac{T_{r'} K_{r'r}}{\sum_i T_i K_{ir}} .
$$

Le dénominateur $(KT)_r = \sum_i T_i K_{ir}$ est l'**accessibilité** de l'acheteur `r` :
la somme, pondérée par les coûts, des avantages comparatifs de tous ses fournisseurs
potentiels.

### 1.3 Ce qu'on observe (et ce qu'on n'observe pas)

On **n'observe pas** les flux bilatéraux `γ_{r'r}`. On observe seulement les **ventes totales**
de chaque région vendeuse `r'`, agrégées sur tous les acheteurs avec un poids `ω_r > 0`
(la taille de marché de l'acheteur `r`) :

$$
\gamma_{r'} = \sum_r \omega_r\, \gamma_{r'r}
            = T_{r'} \sum_r \omega_r \frac{K_{r'r}}{(KT)_r}.
$$

En écriture vectorielle, avec $\oslash$ la division terme à terme :

$$
\boxed{\;\gamma(T) = \operatorname{diag}(T)\, K\, \operatorname{diag}(KT)^{-1}\, \omega\;}
$$

**Le problème d'inversion.** On connaît `K` (via `α`), `ω`, et le vecteur observé `γ`.
On cherche `T`.

### 1.4 Une indétermination inévitable : l'échelle

Multiplier tout `T` par une constante `c > 0` ne change **rien** aux parts :

$$
\frac{c\,T_i K_{ir}}{\sum_j c\,T_j K_{jr}} = \frac{T_i K_{ir}}{\sum_j T_j K_{jr}}
\quad\Longrightarrow\quad \gamma(cT) = \gamma(T).
$$

`T` n'est donc identifié **qu'à une constante multiplicative près**. C'est normal : un
avantage comparatif est une notion *relative*. On lève l'ambiguïté par une **normalisation**,
par exemple

$$
\boxed{\,T_1 = 1\,}
$$

(dans le code, on normalise à la plus grosse région du secteur plutôt qu'à la région 1 —
c'est le même geste). Il reste alors `N − 1` inconnues.

### 1.5 La question précise

> **Conditionnellement à `α` (donc à `K`), et une fois `T_1 = 1` fixé, existe-t-il un
> unique `T > 0` reproduisant les parts observées `γ` ?**

L'intuition naïve serait de compter les équations : `N − 1` équations non linéaires,
`N − 1` inconnues. Mais « autant d'équations que d'inconnues » **ne garantit jamais**
l'unicité pour un système non linéaire. Il faut un vrai argument. C'est l'objet de la
partie 2, et la réponse est **oui**.

---

## 2. La solution : c'est un problème de mise à l'échelle matricielle

### 2.1 L'idée clé — réécrire l'inversion comme un « scaling »

Introduisons un second vecteur positif inconnu, le scaling des colonnes :

$$
s_r = \frac{\omega_r}{(KT)_r} > 0 .
$$

Alors la relation observée se réécrit simplement $\gamma_i = T_i \sum_r K_{ir} s_r$.
Fabriquons la **matrice de flux**

$$
F_{ir} = T_i\, K_{ir}\, s_r, \qquad\text{c.-à-d.}\qquad F = \operatorname{diag}(T)\,K\,\operatorname{diag}(s).
$$

Calculons ses marges (ses sommes en ligne et en colonne) :

$$
\underbrace{\sum_r F_{ir} = T_i \sum_r K_{ir} s_r = \gamma_i}_{\text{marges lignes} = \gamma},
\qquad
\underbrace{\sum_i F_{ir} = s_r \sum_i K_{ir} T_i = s_r (KT)_r = \omega_r}_{\text{marges colonnes} = \omega}.
$$

Deux natures différentes, à bien distinguer :

- La marge **colonne** `= ω` est **automatique** : elle découle de la définition
  $s_r = \omega_r/(KT)_r$ et vaut `ω` *pour tout `T`*. Ce n'est pas une cible qu'on impose,
  c'est l'adding-up des parts ($\sum_i \gamma_{ir} = 1$ pour chaque acheteur).
- La marge **ligne** est la vraie cible — mais seulement **en proportions**. En effet, en
  sommant sur `i` la relation du modèle et en réutilisant l'adding-up,
  $$
  \sum_i \gamma_i(T) = \sum_r \omega_r \underbrace{\sum_i \frac{T_i K_{ir}}{(KT)_r}}_{=\,1}
  = \sum_r \omega_r .
  $$
  **Le modèle produit donc toujours un `γ` de total $\sum_r\omega_r$, quel que soit `T`**
  (c'est le pendant de l'invariance d'échelle du §1.4 : $\gamma(cT)=\gamma(T)$). Le *niveau*
  total de `γ` est verrouillé, pas ciblable ; `T` ne peut reproduire que les **proportions**
  `γ_i / Σγ`.

> **Attention — la compatibilité $\sum_i\gamma_i=\sum_r\omega_r$ n'est pas satisfaite ici.**
> Dans l'application, $\sum_r\omega_r = 1$ (poids `Ê = emp_pi_r` renormalisés) mais
> $\sum_i\gamma_i = \texttt{domestic\_share} < 1$ : une part de la demande fuit vers des
> vendeurs *étrangers* absents de l'ensemble des régions. Ce total manquant **ne porte
> aucune information sur la géométrie `T` région-à-région**. On travaille donc en version
> **projective** : on renormalise la cible à `Σω` (passage aux parts
> $\tilde\gamma_i = \gamma_i\cdot\Sigma\omega/\Sigma\gamma$) — légitime *précisément parce
> que* le scale de `T` est libre. Après ce geste, $\sum_i\tilde\gamma_i = \sum_r\omega_r$
> et le théorème ci-dessous s'applique tel quel.

Ce problème porte un nom : c'est le **problème de mise à l'échelle matricielle**, aussi
appelé **problème de Sinkhorn** (ou RAS chez les économistes, ou « bridge de Schrödinger »
en probabilités). Et pour ce problème, il existe un **théorème d'unicité**.

### 2.2 Le théorème de Sinkhorn

> **Théorème (Sinkhorn, 1967 ; Menon–Schneider).** Soit `K` une matrice à entrées
> strictement positives, et `γ`, `ω` des marges positives *compatibles*
> (i.e. $\sum_i \gamma_i = \sum_r \omega_r$). Alors il existe des vecteurs positifs
> `T`, `s` tels que $F = \operatorname{diag}(T)\,K\,\operatorname{diag}(s)$ ait ces marges,
> et le couple `(T, s)` est **unique à la transformation d'échelle près**
> $(T, s) \mapsto (cT,\; s/c)$.

Deux hypothèses à vérifier dans notre cas.

- **Positivité de `K`** : avec `K_{r'r} = d_{r'r}^{-\alpha\theta}`, toutes les distances
  sont finies, donc toutes les entrées sont strictement positives. ✔ *automatique*.
- **Compatibilité des marges** : $\sum_i\gamma_i = \sum_r\omega_r$. Elle est **fausse
  telle quelle** ($\Sigma\gamma=\texttt{domestic\_share}<1=\Sigma\omega$). On la rétablit
  par la renormalisation projective du §2.1 (remplacer `γ` par
  $\tilde\gamma = \gamma\cdot\Sigma\omega/\Sigma\gamma$), autorisée par la liberté d'échelle
  de `T`. ✔ *après passage aux proportions*.

Les deux hypothèses étant assurées, le théorème donne un couple `(T, s)` unique à l'échelle
près. Cette liberté d'échelle `c` est exactement l'indétermination de la partie 1.4 : en
fixant `T_1 = 1` on la supprime, et il reste un **`T` unique**.

**Réponse à la question de la partie 1.5 : oui, `T` est unique, et l'unicité est *globale*.**
Ce n'est pas seulement « localement, si le jacobien est de plein rang » — c'est vrai partout.

### 2.3 Une preuve auto-suffisante (par la convexité)

Le théorème de Sinkhorn peut sembler tomber du ciel. En voici une preuve courte et
élémentaire, qui a le mérite d'être *constructive* (elle dira aussi comment calculer `T`).

Posons le changement de variable $a_i = \log T_i$ et considérons la fonction

$$
\boxed{\;g(a) = \sum_r \omega_r \log\!\Big(\sum_i e^{a_i} K_{ir}\Big) \;-\; \sum_i \gamma_i\, a_i\;}
$$

C'est une fonction de `N` variables réelles, sans contrainte.

**Étape 1 — les points critiques de `g` sont exactement les solutions de l'inversion.**
On calcule la dérivée partielle (en se rappelant que $(KT)_r = \sum_i e^{a_i} K_{ir}$) :

$$
\frac{\partial g}{\partial a_i}
= \sum_r \omega_r \frac{e^{a_i} K_{ir}}{(KT)_r} - \gamma_i
= T_i \sum_r \omega_r \frac{K_{ir}}{(KT)_r} - \gamma_i .
$$

Annuler ce gradient donne précisément $\gamma_i = T_i \sum_r \omega_r K_{ir}/(KT)_r$ :
**c'est notre équation d'inversion.** Résoudre l'inversion ⟺ trouver un point critique de `g`.

**Étape 2 — `g` est convexe.** Le terme $\sum_r \omega_r \log(\sum_i e^{a_i} K_{ir})$ est
une somme (à poids positifs `ω_r`) de fonctions *log-sum-exp* composées avec des applications
linéaires : c'est un grand classique, c'est convexe. Le terme $-\sum_i \gamma_i a_i$ est
linéaire, donc convexe aussi. Donc `g` est convexe. Pour une fonction convexe, **tout point
critique est un minimum global** : l'existence d'une solution de l'inversion équivaut à
l'existence d'un minimiseur de `g`.

> **Compatibilité et bornitude.** Le long de la direction d'échelle $a \mapsto a + c\mathbf 1$
> on a $g(a+c\mathbf 1) = g(a) + c\,(\Sigma\omega - \Sigma\gamma)$. Si $\Sigma\gamma\neq\Sigma\omega$,
> `g` est donc **non bornée** (elle file vers $-\infty$ dans cette direction) : pas de
> minimiseur global — c'est la traduction analytique de « le total de `γ` n'est pas ciblable ».
> On utilise donc la cible renormalisée $\tilde\gamma$ (avec $\Sigma\tilde\gamma=\Sigma\omega$,
> §2.1) ; alors le terme d'échelle s'annule, `g` est plate dans la seule direction $\mathbf 1$
> et strictement convexe partout ailleurs (Étape 3), d'où un minimiseur unique sur la tranche
> `T_1 = 1`. (Minimiser `g` avec la cible brute `γ` sur cette tranche « marcherait » aussi
> numériquement, mais ferait absorber tout le déséquilibre `Σω − Σγ` par la seule région de
> référence — un choix asymétrique ; la renormalisation répartit proprement l'écart en
> proportions, ce que fait aussi l'itération Sinkhorn du §3.)

**Étape 3 — `g` est *strictement* convexe, sauf dans une seule direction.**
La hessienne de `g` vaut

$$
\nabla^2 g(a) = \sum_r \omega_r \big(\operatorname{diag}(p^r) - p^r (p^r)^\top\big),
\qquad p^r_i = \frac{T_i K_{ir}}{(KT)_r} .
$$

Pour chaque `r`, le vecteur `p^r` est un **vecteur de probabilité** ($p^r_i \ge 0$,
$\sum_i p^r_i = 1$) : c'est la loi des parts bilatérales de l'acheteur `r`. La matrice
$\operatorname{diag}(p^r) - p^r (p^r)^\top$ est la **matrice de covariance** de la loi
catégorielle `p^r` : elle est semi-définie positive, et son noyau est exactement la droite
$\operatorname{vect}\{\mathbf 1\}$ (le vecteur constant). En sommant sur `r` avec des poids
`ω_r > 0` et grâce à la **stricte positivité de `K`** (toutes les parts sont non nulles),
le noyau de la somme reste *uniquement* $\operatorname{vect}\{\mathbf 1\}$.

Or la direction $\mathbf 1$ correspond, en variable $a = \log T$, à $T \mapsto cT$ :
**c'est exactement l'indétermination d'échelle.** La fonction `g` est donc **strictement
convexe modulo l'échelle** : dans le sous-espace orthogonal à $\mathbf 1$ (par exemple la
tranche `T_1 = 1`, i.e. `a_1 = 0`), elle est strictement convexe, donc son minimiseur y est
**unique**.

Conclusion : il existe un unique `T` avec `T_1 = 1` résolvant l'inversion. **CQFD.**

### 2.4 Ce qu'il faut retenir (et deux malentendus à éviter)

- **C'est la positivité de `K` qui fait tout**, pas la symétrie. On a écrit `K` symétrique
  parce que la distance l'est, mais la preuve n'utilise jamais `K = Kᵀ`. Le résultat vaut
  pour une matrice `K` rectangulaire non symétrique (utile : voir partie 3.2).
- **Les « cas dégénérés » ne cassent pas l'unicité.** On pourrait craindre que des lignes
  de `K` proportionnelles, ou une matrice `K` presque de rang 1, produisent plusieurs
  solutions. **Faux.** Tant que `K > 0`, la stricte convexité modulo l'échelle tient, quel
  que soit le rang de `K`. Une matrice de rang 1, $K = ab^\top$ avec $a, b > 0$, admet un
  scaling *unique*.
- **Unicité ≠ identification forte.** Ce qui *peut* mal se passer en pratique, ce n'est pas
  la non-unicité, c'est un **mauvais conditionnement** : si deux régions ont des profils de
  distance très proches, la hessienne $\nabla^2 g$ a de petites valeurs propres. La solution
  reste unique, mais elle est *sensible au bruit* (grandes erreurs-types, convergence lente).
  C'est une **identification faible**, pas une absence d'identification. Dans le code, c'est
  précisément ce que surveille `screen_T_identification` (la plus petite valeur propre du
  bloc `H[T,T]`).

---

## 3. Le lien avec le code : Sinkhorn, et faut-il faire autrement ?

### 3.1 L'algorithme de Sinkhorn

La preuve de la partie 2.3 suggère un algorithme, et c'est l'algorithme historique de
Sinkhorn : **corriger alternativement les marges**.

Partant de `T > 0` quelconque (par exemple `T = (1, …, 1)`) :

```
Répéter :
    Phi = Kᵀ T          # marges colonnes courantes  (accessibilités)
    s   = ω ⊘ Phi        # corrige les colonnes : les sommes-colonnes valent ω
    M   = K s            # ce que deviennent les marges lignes
    T   = γ ⊘ M          # corrige les lignes : les sommes-lignes valent γ
    T   = T / T_1        # normalisation d'échelle (ne change pas γ)
jusqu'à convergence
```

En substituant, la mise à jour de `T` est en une ligne :

$$
\boxed{\;T_i^{(n+1)} = \frac{\gamma_i}{\displaystyle\sum_r K_{ir}\,\dfrac{\omega_r}{(KT^{(n)})_r}}\;}
$$

**Pourquoi ça converge ?** Chaque demi-pas est une *projection* qui remet une marge à sa
valeur cible. Le théorème de Birkhoff dit qu'une application linéaire à noyau positif est
une **contraction dans la métrique projective de Hilbert** (une distance qui « oublie »
l'échelle), avec un taux de contraction strictement `< 1` dès que `K > 0`. Les divisions
par `γ` et `ω` sont des isométries pour cette métrique. Le pas complet est donc une
contraction : `T^{(n)}` converge géométriquement vers l'unique `T★` normalisé. La vitesse
est d'autant meilleure que `K` est « bien mélangée » (régions aux profils de distance
contrastés) — le même contraste qui garantit le bon conditionnement de la partie 2.4.

### 3.2 L'initialisation du code *est* du Sinkhorn (amorti)

**Oui : l'initialisation de `T` conditionnelle à `α` implémentée dans le code est bien
l'algorithme de Sinkhorn.** La fonction `invert_T_from_gamma` (`load_parameters.jl`) fait
tourner exactement l'itération ci-dessus, avec `ω = Ê` (la taille de marché observée
`emp_pi_r`) et `K = τ^{-θ}` évalué au *prior* `α`. Trois détails d'implémentation, aucun ne
change le fond :

1. **Amortissement en espace log.** Au lieu de $T \leftarrow \gamma/M$ sec, le code fait
   $$
   \log T_r^{(n+1)} = (1-\delta)\log T_r^{(n)} + \delta \log\!\big(\gamma_r/M_r\big), \qquad \delta = 0.5.
   $$
   C'est un pas de Sinkhorn *relaxé* : $\delta = 1$ redonne le Sinkhorn pur. L'amortissement
   robustifie la convergence (utile quand `K` est mal conditionnée) au prix d'un peu de
   vitesse.
2. **Matrice rectangulaire.** Dans le vrai modèle, les *vendeurs* `r` et les *acheteurs* `dr`
   (destinations aval) ne sont pas le même ensemble : `K` est de taille
   `(R_full × R_downstream)`, **non carrée**. `Φ` vit du côté destinations
   (`Φ = Kᵀ T`) et `M` du côté vendeurs (`M = K s`). C'est du Sinkhorn *rectangulaire* ;
   comme la partie 2.4 le souligne, la symétrie n'était pas nécessaire, donc l'unicité tient
   toujours. (La simplification « `K` symétrique ⇒ `Φ = KT` » ne vaut, elle, que pour le
   modèle-jouet carré.)
3. **Normalisation à la région de référence**, secteur par secteur, plutôt qu'à la région 1 ;
   et **initialisation aux parts observées** `T[r] = γ_{r}` plutôt qu'à `(1,…,1)`. Un bon
   point de départ, plus proche de la solution, donc moins d'itérations.

Autrement dit : **on ne peut pas faire « mieux que Sinkhorn » ici, parce que le code fait déjà
Sinkhorn.** La question n'est pas Sinkhorn *vs* autre chose, mais : ce Sinkhorn est-il le bon
outil, ou vaudrait-il mieux minimiser directement `g` ?

### 3.3 Sinkhorn ou la forme convexe directe ?

Les deux routes mènent au **même** unique `T★` (ce sont deux façons de résoudre
$\nabla g = 0$). Le choix est purement numérique.

| Critère | Sinkhorn (le code) | Minimisation directe de `g` (L-BFGS / Newton) |
|---|---|---|
| Ce que c'est | descente de coordonnées par blocs sur `g` | descente/Newton sur `g` entière |
| Coût par itération | très faible : deux produits matrice–vecteur | gradient idem ; Newton demande la hessienne `N×N` |
| Vitesse de convergence | **linéaire** (taux = contraction de Birkhoff de `K`) | Newton : **quadratique** près de l'optimum |
| Robustesse | excellente si `K` bien mélangée ; **lente** si `K` mal conditionnée | Newton reste rapide même mal conditionné, mais plus lourd/fragile à coder |
| Simplicité | triviale (4 lignes, pas de dérivées) | il faut coder gradient (+ hessienne pour Newton) |

**Recommandation, et pourquoi c'est le bon choix ici.**

- Pour une **initialisation**, on ne cherche pas la solution à `1e-12` près : on veut un
  point de départ raisonnable pour l'optimiseur global (PSO/CMA-ES) qui suit. Sinkhorn, simple
  et sans dérivées, atteint largement cette précision en quelques dizaines d'itérations.
  C'est le bon niveau d'effort. **Rien à changer.**
- La forme convexe `g` reste utile pour **deux** raisons, même si on garde Sinkhorn en
  production :
  1. C'est **la preuve d'unicité** (partie 2.3) — la justification théorique de toute
     l'opération.
  2. C'est le **plan de secours**. Si un jour une géométrie très mal conditionnée (régions
     aux distances quasi identiques) fait ramer Sinkhorn — beaucoup d'itérations, ou stagnation
     avant la tolérance — on peut basculer sur **Newton sur `g`** : la convergence quadratique
     ignore le mauvais conditionnement (au prix d'un solve linéaire `N×N` par pas). Comme `g`
     est convexe, Newton (amorti par recherche linéaire) converge globalement, sans risque de
     minimum local.

En résumé : **Sinkhorn est le bon défaut**, et c'est bien ce que le code fait ; **la forme
convexe est le garde-fou** — on la garde en tête comme preuve et comme solveur de repli, on
n'y bascule que si le conditionnement l'exige.

---

## 4. Résumé

1. **Conditionner sur `α` fixe le noyau `K`** : l'inversion des avantages comparatifs devient
   un problème sur une matrice connue.
2. Ce problème est **exactement** une mise à l'échelle matricielle : trouver `T` (et un
   scaling `s`) tels que $\operatorname{diag}(T)\,K\,\operatorname{diag}(s)$ ait pour marges
   `γ` (ventes observées) et `ω` (tailles de marché).
3. Comme `K = d^{-\alpha\theta} > 0`, le **théorème de Sinkhorn** garantit un `T` **unique**
   à l'échelle près ; la normalisation `T_1 = 1` (ou à la région de référence) le fixe
   complètement. L'unicité est **globale**, prouvée par la stricte convexité (modulo échelle)
   du potentiel `g`. Comme $\Sigma\gamma=\texttt{domestic\_share}\neq\Sigma\omega=1$ (fuite vers
   l'étranger), on travaille en version **projective** : seules les *proportions* de `γ_ls` sont
   ciblables — le total est verrouillé à `Σω` par l'adding-up — donc on renormalise la cible à
   `Σω`, ce que l'itération Sinkhorn (renormalisation par passe) fait automatiquement.
4. La non-unicité **n'arrive jamais** pour `K > 0` ; le seul risque pratique est une
   **identification faible** (mauvais conditionnement), pas une ambiguïté de solution.
5. L'initialisation du code (`invert_T_from_gamma`) **est** l'algorithme de Sinkhorn (amorti,
   rectangulaire, normalisé par secteur) : c'est le bon outil, à garder. La **forme convexe
   `g`** sert de preuve d'unicité et de solveur de secours (Newton) pour les cas très mal
   conditionnés.

---

### Références

- R. Sinkhorn (1967), *Diagonal equivalence to matrices with prescribed row and column sums*.
- M. V. Menon, H. Schneider (1969), sur l'unicité de la mise à l'échelle des matrices positives.
- G. Birkhoff (1957), contraction dans la métrique projective de Hilbert.
- Voir aussi `documentation/model.md` (le modèle EK complet) et `documentation/inference.md`
  (jacobien, erreurs-types, diagnostic d'identification `screen_T_identification`).
