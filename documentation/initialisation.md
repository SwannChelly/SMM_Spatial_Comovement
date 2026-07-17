# Initialisation de T : inverser les avantages comparatifs à partir des parts

**But.** Une fois qu'on se donne un *prior* sur le paramètre de coût commercial `α`, on peut
retrouver **de façon unique** le vecteur des avantages comparatifs `T`, à partir des seules
parts d'approvisionnement observées. Cette note explique pourquoi (partie 1–2) et comment le
code le fait (partie 3). Niveau visé : prépa (algèbre linéaire, suites récurrentes, convexité).

Code concerné : `invert_T_from_gamma` dans `load_parameters.jl` (SECTION 10b).

---

## 1. Le problème

### 1.1 Le décor

`N` régions. Chaque région vendeuse `r'` a un **avantage comparatif** `T_{r'} > 0` (sa
productivité fondamentale à la Eaton–Kortum). Le coût de transport de `r'` vers un acheteur
`r` est une loi de puissance de la distance, $\tau_{r'r} = d_{r'r}^{\,\alpha}$. On résume la
géographie par le **noyau de coût**

$$
K_{r'r} = \tau_{r'r}^{-\theta} = d_{r'r}^{-\alpha\theta} \;>\; 0
$$

(`θ` = paramètre de Fréchet, `θ = 1` dans le code). **Dès que `α` est fixé, `K` est
entièrement connu** — c'est tout l'intérêt de conditionner sur `α` : l'inversion devient un
problème sur une matrice fixe. Le modèle EK donne la part des achats de `r` venant de `r'` :

$$
\gamma_{r'r} = \frac{T_{r'} K_{r'r}}{\sum_i T_i K_{ir}}, \qquad (KT)_r = \sum_i T_i K_{ir}\ \text{(accessibilité de } r).
$$

On **n'observe pas** ces flux bilatéraux, seulement les **ventes totales** de chaque vendeur,
agrégées sur les acheteurs avec un poids `ω_r > 0` (leur taille de marché) :

$$
\boxed{\;\gamma_{r'}(T) = \sum_r \omega_r\, \gamma_{r'r} = T_{r'} \sum_r \omega_r \frac{K_{r'r}}{(KT)_r}\;}
$$

On connaît `K` (via `α`), `ω` et le vecteur observé `γ` ; **on cherche `T`.**

### 1.2 Les deux normalisations (à garder en tête dès le départ)

L'objet `γ` n'est pas un vecteur brut : deux conventions le déterminent, et elles gouvernent
toute la suite.

**(N1) `T` est normalisé par une région de référence.** Multiplier `T` par une constante
`c > 0` ne change rien aux parts :

$$
\frac{c\,T_i K_{ir}}{\sum_j c\,T_j K_{jr}} = \frac{T_i K_{ir}}{\sum_j T_j K_{jr}}
\quad\Longrightarrow\quad \gamma(cT) = \gamma(T).
$$

`T` n'est donc identifié **qu'à une constante multiplicative près** (un avantage comparatif est
relatif). On lève l'ambiguïté en fixant, **par secteur**, la région de référence à 1 :

$$
\boxed{\,T_{\text{ref}} = 1\,}\qquad(\text{dans le code : } \texttt{T\_REF\_REGION[s]}).
$$

Il reste `N − 1` inconnues.

**(N2) `γ` est repondéré par la part domestique.** Les parts observées ne somment **pas** à 1 :
une partie de l'approvisionnement vient de vendeurs *hors du système de régions* (l'étranger).
Les `γ_r` sont les parts *intérieures*, c.-à-d. les parts normalisées (qui, elles, sommeraient
à 1) **repondérées par la part domestique** `s_dom < 1` :

$$
\sum_r \gamma_r = s_{\text{dom}} < 1, \qquad\text{alors que}\qquad \sum_r \omega_r = 1 .
$$

Retenez ce déséquilibre $\sum\gamma \neq \sum\omega$ : il ressort en partie 2 et il est
inoffensif (le total de `γ` ne porte aucune information sur `T`, seul le *profil* compte).

### 1.3 La question précise

> **Conditionnellement à `α` (donc à `K`), et une fois `T_{\text{ref}} = 1` fixé, existe-t-il
> un unique `T > 0` reproduisant le profil des parts observées `γ` ?**

Compter les équations (`N − 1` équations, `N − 1` inconnues) ne suffit pas : pour un système
non linéaire, ça ne garantit jamais l'unicité. Il faut un argument. Réponse : **oui.**

---

## 2. La solution : une mise à l'échelle matricielle (Sinkhorn)

### 2.1 Réécrire l'inversion comme un « scaling »

Introduisons un vecteur positif auxiliaire, le **scaling des colonnes** :

$$
s_r = \frac{\omega_r}{(KT)_r} > 0 .
$$

La relation observée devient $\gamma_i = T_i \sum_r K_{ir}\, s_r$. Formons la **matrice de flux**
$F_{ir} = T_i K_{ir} s_r$, soit $F = \operatorname{diag}(T)\,K\,\operatorname{diag}(s)$, et
regardons ses sommes en ligne et en colonne :

$$
\sum_r F_{ir} = T_i \sum_r K_{ir} s_r = \gamma_i
\qquad(\text{marge ligne}),
\qquad\qquad
\sum_i F_{ir} = s_r (KT)_r = \omega_r
\qquad(\text{marge colonne}).
$$

Ces deux marges n'ont pas le même statut :

- La **colonne** vaut `ω` **automatiquement**, pour tout `T` (c'est la définition de `s_r`, i.e.
  l'adding-up des parts $\sum_i \gamma_{ir} = 1$). Ce n'est pas une cible qu'on impose.
- La **ligne** est la vraie cible `γ`, mais **seulement en proportions**. En sommant sur `i` :
  $$
  \sum_i \gamma_i(T) = \sum_r \omega_r \underbrace{\sum_i \tfrac{T_i K_{ir}}{(KT)_r}}_{=\,1}
  = \sum_r \omega_r = 1 .
  $$
  Le modèle produit donc **toujours** un `γ` de total 1, quel que soit `T` (c'est le pendant de
  l'invariance d'échelle (N1)). Or `γ` observé a pour total `s_dom < 1` (N2). Le niveau total est
  donc **verrouillé, non ciblable** ; `T` ne peut reproduire que le profil `γ_r / \sum\gamma`.

**Version projective.** On travaille donc à cible renormalisée
$\tilde\gamma = \gamma \cdot (\sum\omega / \sum\gamma) = \gamma / s_{\text{dom}}$, qui vérifie
$\sum\tilde\gamma = \sum\omega$. C'est légitime **précisément parce que** le scale de `T` est
libre (N1) : matcher `γ` « à une constante près » = matcher `γ̃`.

> **Dans le code.** Cette cible balancée est le const `emp_gamma_ls_tilde` =
> `emp_gamma_ls ./ domestic_share` (`load_parameters.jl`, SECTION 3, $\sum_r = 1$). C'est la
> cible par défaut de l'inversion d'initialisation `invert_T_from_gamma` **et** de l'inversion
> de profilage `invert_T_ge` (`profiling.jl`). La jauge de référence $T_{\text{ref}}=1$ rend le
> `T` retourné identique à celui qu'on obtiendrait avec la `γ` brute, mais la précondition de
> compatibilité des marges de Sinkhorn est ainsi **explicitement** satisfaite.

**Conclusion de 2.1.** Trouver `T` = trouver deux rééchelonnements diagonaux positifs
$\operatorname{diag}(T)$, $\operatorname{diag}(s)$ qui envoient la matrice **fixe et positive**
`K` sur une matrice `F` aux marges $(\tilde\gamma, \omega)$. C'est le **problème de Sinkhorn**
(alias RAS, ou *matrix scaling*).

### 2.2 Le théorème de Sinkhorn

> **Théorème (Sinkhorn 1967 ; Menon–Schneider 1969).** Soit `K` à entrées **strictement
> positives** et deux marges positives **compatibles** ($\sum_i \tilde\gamma_i = \sum_r \omega_r$).
> Alors il existe des vecteurs positifs `T`, `s` tels que
> $\operatorname{diag}(T)\,K\,\operatorname{diag}(s)$ ait ces marges, **uniques à l'échelle près**
> $(T,s)\mapsto(cT, s/c)$.

Les deux hypothèses sont satisfaites ici :

- **Positivité** : $K_{r'r} = d_{r'r}^{-\alpha\theta} > 0$ (distances finies). ✔ *automatique*.
- **Compatibilité** : rétablie par la renormalisation projective de 2.1
  ($\tilde\gamma = \gamma/s_{\text{dom}}$). ✔ *après passage au profil*.

La liberté d'échelle `c` est exactement (N1) : en fixant $T_{\text{ref}} = 1$ on la supprime,
et il reste **un `T` unique**. L'unicité est **globale** (partout), pas seulement locale. On le
prouve directement en 2.3.

### 2.3 Preuve directe par convexité (explicite)

Posons $a_i = \log T_i$ et considérons la fonction de `N` variables réelles, sans contrainte,

$$
\boxed{\;g(a) = \sum_r \omega_r \log\!\Big(\underbrace{\textstyle\sum_i e^{a_i} K_{ir}}_{\Phi_r}\Big)
       \;-\; \sum_i \tilde\gamma_i\, a_i\;}
$$

**Étape 1 — le gradient de `g` est exactement le résidu de l'inversion.**
Pour dériver le premier terme, on utilise $\dfrac{\partial}{\partial a_i}\log\Phi_r
= \dfrac{1}{\Phi_r}\dfrac{\partial \Phi_r}{\partial a_i}
= \dfrac{e^{a_i}K_{ir}}{\Phi_r}$. D'où

$$
\frac{\partial g}{\partial a_i}
= \sum_r \omega_r \frac{e^{a_i} K_{ir}}{\Phi_r} - \tilde\gamma_i
= \underbrace{T_i \sum_r \omega_r \frac{K_{ir}}{(KT)_r}}_{=\ \gamma_i(T)\ \text{(ventes du modèle)}} - \tilde\gamma_i .
$$

Annuler le gradient, c'est exactement $\gamma_i(T) = \tilde\gamma_i$ : **résoudre l'inversion
⟺ trouver un point critique de `g`.**

**Étape 2 — `g` est convexe.** $\Phi_r = \sum_i e^{a_i}K_{ir}$ est une somme d'exponentielles, et
$a \mapsto \log\sum_i e^{a_i}K_{ir}$ (un *log-sum-exp*) est convexe — résultat classique. Somme à
poids `ω_r > 0` de fonctions convexes, plus un terme linéaire $-\sum_i\tilde\gamma_i a_i$ : donc
`g` est **convexe**. Pour une fonction convexe, tout point critique est un **minimum global** :
l'inversion a une solution ⟺ `g` a un minimiseur.

**Étape 3 — `g` est *strictement* convexe, sauf dans la direction d'échelle.**
On dérive une seconde fois. Notons $p^r_i = \dfrac{e^{a_i}K_{ir}}{\Phi_r}$ (avec $p^r_i \ge 0$,
$\sum_i p^r_i = 1$ : c'est la loi des parts bilatérales de l'acheteur `r`). Alors

$$
\frac{\partial^2 g}{\partial a_i \partial a_j}
= \sum_r \omega_r \big(\delta_{ij}\, p^r_i - p^r_i p^r_j\big),
\qquad\text{soit}\qquad
\nabla^2 g = \sum_r \omega_r\,\big(\operatorname{diag}(p^r) - p^r (p^r)^\top\big).
$$

Chaque bloc $\operatorname{diag}(p^r) - p^r(p^r)^\top$ est la **matrice de covariance** de la loi
catégorielle `p^r` : elle est semi-définie positive (pour tout vecteur `v`,
$v^\top(\operatorname{diag}(p^r)-p^rp^{r\top})v = \operatorname{Var}_{p^r}(v) \ge 0$), et cette
variance s'annule ssi `v` est constant, i.e. son noyau est exactement $\operatorname{vect}\{\mathbf 1\}$.
En sommant sur `r` (poids `ω_r > 0`) et comme **`K > 0`** rend toutes les parts non nulles, le
noyau de $\nabla^2 g$ reste **uniquement** $\operatorname{vect}\{\mathbf 1\}$.

Or $\mathbf 1$ en variable $a = \log T$ correspond à $T \mapsto cT$ : **c'est l'indétermination
d'échelle (N1)**. Donc `g` est **strictement convexe modulo l'échelle**. Sur la tranche
$T_{\text{ref}} = 1$ (i.e. $a_{\text{ref}} = 0$), qui coupe transversalement cette unique
direction plate, `g` est strictement convexe : **son minimiseur y est unique.**

**Conclusion.** Il existe un unique `T` avec $T_{\text{ref}} = 1$ résolvant l'inversion. ∎

### 2.4 Deux mises en garde

- **C'est la positivité de `K` qui fait tout, pas la symétrie** (la preuve n'utilise jamais
  $K = K^\top$). Le résultat vaut donc pour `K` **rectangulaire** — utile, car c'est le cas du
  code (partie 3.2). Les « cas dégénérés » redoutés (lignes de `K` proportionnelles, `K` de
  rang 1) **ne cassent pas** l'unicité : tant que `K > 0`, la stricte convexité modulo échelle
  tient, quel que soit le rang.
- **Unicité ≠ identification forte.** Le seul vrai risque pratique n'est pas la non-unicité,
  mais le **mauvais conditionnement** : si deux régions ont des profils de distance quasi
  identiques, $\nabla^2 g$ a de petites valeurs propres → solution unique mais sensible au bruit
  (grandes erreurs-types, convergence lente). C'est une *identification faible*. Le code la
  surveille via `screen_T_identification` (plus petite valeur propre du bloc `H[T,T]`).

---

## 3. L'algorithme d'initialisation

### 3.1 L'itération de Sinkhorn

L'idée : **corriger alternativement les deux marges**. Partant de `T > 0` :

```
Répéter :
    Φ = Kᵀ T           # marges colonnes (accessibilités des acheteurs)
    s = ω ⊘ Φ          # corrige les colonnes  → sommes-colonnes = ω
    M = K s            # marges lignes induites
    T = γ ⊘ M          # corrige les lignes    → sommes-lignes ∝ γ
    T = T / T_ref      # normalisation d'échelle (N1) — ne change pas le profil
jusqu'à convergence
```

En substituant, la mise à jour de `T` tient en une ligne :

$$
\boxed{\;T_i^{(n+1)} = \frac{\gamma_i}{\displaystyle\sum_r K_{ir}\,\dfrac{\omega_r}{(KT^{(n)})_r}}\;}
$$

**Pourquoi ça converge.** Le théorème de Birkhoff dit qu'une application linéaire à noyau
positif est une **contraction dans la métrique projective de Hilbert** (une distance qui oublie
l'échelle), de taux `< 1` dès que `K > 0` ; les divisions par `γ` et `ω` en sont des isométries.
Le pas complet est donc une contraction : `T^{(n)}` converge géométriquement vers l'unique `T★`.
La renormalisation par passe (`T = T / T_ref`) fait qu'à la limite le modèle reproduit le
**profil** de `γ` (total mécaniquement égal à `Σω`, cf. 2.1) — la version projective, sans avoir
à renormaliser explicitement la cible.

### 3.2 Ce que fait exactement `invert_T_from_gamma`

La fonction est **précisément cette itération de Sinkhorn**, appliquée **secteur par secteur**,
avec `ω = Ê` (taille de marché `emp_pi_r`, renormalisée à 1) et `K = τ^{-θ}` évalué au prior `α`.
Voici l'algorithme fidèle au code (`load_parameters.jl:461-511`), avec ses trois écarts au
Sinkhorn scolaire — aucun ne change la limite :

```
Entrées : prior α ; par secteur s : régions actives R_s, région de référence ref = T_REF_REGION[s]
Constantes : K[r,dr] = max(d[r,dr], 1)^(-θα)      # (R_full × R_downstream), power-law au prior α
             Ê[dr]  = emp_pi_r[dr] / Σ emp_pi_r    # poids acheteurs (downstream), somme = 1
             γ[r]   = emp_gamma_ls[r,s]            # parts observées, somme = s_dom (N2)
             δ = 0.5 (amortissement), tol = 1e-11, max_iter = 1000

Pour chaque secteur s :
    # init aux parts observées, normalisées à la référence
    T[r] ← max(γ[r], 1e-12) pour r ∈ R_s ;   T[r] ← T[r] / T[ref]

    Répéter (≤ max_iter) :
        Φ[dr] ← Σ_{r∈R_s} T[r] · K[r,dr]                  # marge colonne  (= Kᵀ T)
        Pour r ∈ R_s :
            M[r]  ← Σ_dr K[r,dr] · Ê[dr] / Φ[dr]          # marge ligne induite (= K s, s = Ê⊘Φ)
            Tr    ← γ[r] / M[r]                            # pas de Sinkhorn brut
            T⁺[r] ← exp( (1−δ)·log T[r] + δ·log Tr )       # (a) amorti en espace log
        T[r] ← T⁺[r] / T⁺[ref]  pour r ∈ R_s              # renormalisation à la référence (N1)
        arrêt si max_r | log T[r] − log T_précédent[r] | < tol
```

Les trois écarts :

- **(a) Amortissement en espace log** ($δ = 0.5$) : au lieu de $T \leftarrow \gamma/M$ sec, on fait
  une moyenne géométrique $T^{(n+1)} = (T^{(n)})^{1-δ}\,(\gamma/M)^{δ}$. C'est un pas de Sinkhorn
  *relaxé* ($δ = 1$ redonne le Sinkhorn pur) : plus robuste quand `K` est mal conditionnée, au
  prix d'un peu de vitesse.
- **(b) Noyau rectangulaire** : vendeurs `r` (les `R_full`) et acheteurs `dr` (les
  `R_downstream`) sont deux ensembles distincts, donc `K` **n'est pas carrée**. `Φ` vit côté
  acheteurs, `M` côté vendeurs. La partie 2.4 garantit que l'unicité tient quand même (seule
  la positivité comptait). La simplification « `K` symétrique ⇒ `Φ = KT` » ne vaut que pour le
  modèle-jouet carré de la partie 1.
- **(c) Bon départ** : on initialise aux parts observées `T[r] = γ[r]` (plus proche de la
  solution que `(1,…,1)`), normalisées à la référence — d'où peu d'itérations en pratique.

Autrement dit : **le code fait déjà du Sinkhorn.** On ne peut pas faire « mieux que Sinkhorn » ;
la seule vraie alternative est la forme convexe directe (ci-dessous).

### 3.3 Sinkhorn ou minimisation directe de `g` ?

Les deux routes donnent le **même** `T★` (deux façons de résoudre $\nabla g = 0$). Le choix est
purement numérique.

| | Sinkhorn (le code) | Minimiser `g` (L-BFGS / Newton) |
|---|---|---|
| Nature | descente par blocs sur `g` | descente / Newton sur `g` entière |
| Coût / itération | deux produits matrice–vecteur | gradient idem ; Newton : hessienne `N×N` |
| Convergence | **linéaire** (taux de Birkhoff de `K`) | Newton : **quadratique** près de l'optimum |
| Mauvais cond. | lente | Newton : rapide même mal conditionné |
| Simplicité | triviale, sans dérivées | il faut coder gradient (+ hessienne) |

**Recommandation.** Pour une **initialisation** (on ne vise pas `1e-12`, juste un bon point de
départ pour le PSO/CMA-ES qui suit), Sinkhorn — simple, sans dérivées, convergent en quelques
dizaines d'itérations — est le bon outil : **rien à changer.** La forme convexe `g` reste utile
comme (i) **preuve d'unicité** (2.3) et (ii) **plan de secours** : si une géométrie très mal
conditionnée faisait ramer Sinkhorn, on basculerait sur **Newton sur `g`** (convergence
quadratique, insensible au conditionnement ; convexe ⇒ pas de minimum local parasite).

---

## 4. Résumé

1. **Conditionner sur `α` fixe `K`** : l'inversion devient un problème sur une matrice connue.
2. Deux normalisations cadrent `γ` : **(N1)** `T` est défini à l'échelle près, fixée par
   $T_{\text{ref}} = 1$ ; **(N2)** `γ` est repondéré par la part domestique, donc
   $\sum\gamma = s_{\text{dom}} < 1 = \sum\omega$ — seul le *profil* de `γ` est ciblable.
3. L'inversion est **exactement** un *matrix scaling* (Sinkhorn). Comme `K > 0`, le théorème
   garantit un `T` **unique** à l'échelle près ; $T_{\text{ref}} = 1$ le fixe. L'unicité est
   **globale**, prouvée par la stricte convexité (modulo échelle) du potentiel `g`.
4. La non-unicité **n'arrive jamais** pour `K > 0` ; le seul risque est l'**identification
   faible** (mauvais conditionnement), pas l'ambiguïté.
5. `invert_T_from_gamma` **est** l'algorithme de Sinkhorn (amorti, rectangulaire, par secteur,
   démarré aux parts observées) : le bon outil, à garder. La **forme convexe `g`** sert de
   preuve et de solveur de secours (Newton).

---

### Références

- R. Sinkhorn (1967), *Diagonal equivalence to matrices with prescribed row and column sums*.
- M. V. Menon, H. Schneider (1969), unicité de la mise à l'échelle des matrices positives.
- G. Birkhoff (1957), contraction dans la métrique projective de Hilbert.
- Voir aussi `documentation/model.md` (modèle EK complet) et `documentation/inference.md`
  (jacobien, erreurs-types, `screen_T_identification`).
