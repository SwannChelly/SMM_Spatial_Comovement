

---------

# Meilleure estimation de $\alpha$

Je spécifie $\tau_{r'r} = d_{r'r}^\alpha$. Dernièrement j'ai eu une meilleure estimation de $\alpha$ car j'ai utilisé un meilleur prior sur sa valeur. Pour se faire, il faut qu'on comprenne ce que l'on fait dans la régression à la marge extensive du point de vue du modèle. Dans le modèle, un fournisseur $z$ de $(r',s)$ fourni à $r$ avec la probabilité

$$\rho_{r'rs}(z) = e^{-(T_{r's} \gamma_{r'rs}^{-1}z^{-\theta})}$$

Quand on intègre ça sur toute la distribution de productivité $z$ (loi de fréchet de paramètre $T_{rs}$) on obtient la probabilité qu'une entreprise de $(r',s)$ soit sélectionnée par $r$ comme fournisseur $\gamma_{r'rs} = T_{r's} \tau_{r'r}^{-\theta}/\Phi_{rs}$. 


Théoriquement, si on prend la cellule $(r',s)$ comme référence et que l'on note $W_{r} | z$ l'événement correspondant au fait que l'entreprise de cette cellule de productivité $z$ remporte le marché $r$ alors, la probabilité que cette entreprise fournisse à l'industrie downstream est $\mathbb{P}(\cup_r W_r | z )$. 

$$
\tilde{\rho}_{r's}(z) = \mathbb{P}(\cup_r W_r | z )
$$
Dans le papier on note dans l'appendix que cette probabilité s'écrit : 

$$\begin{equation}
    \tilde{\rho}_{r's}(z)\equiv 1-\prod_{r=1}^R (1-\rho_{r'rs}(z))
\end{equation}$$

Où $\rho_{r'rs}(z)$ est la probabilité que $z$ serve $r$. Mais je ne suis pas sûr de ça. La probabilité qu'une entreprise soit sélectionnée par deux régions $r_1$ et $r_2$ n'est pas égale au produit des probabilités: une entreprise avec une forte productivité a plus de change de gagner les deux marchés. L'expression exacte de $\tilde{\rho}_{r's}(z)$ est la suivante : 

$$
\tilde{\rho}_{r's}(z) = \mathbb{P}(\cup_r W_r | z ) = \sum_{\emptyset\neq S\subseteq\mathcal{D}} (-1)^{|S|+1}\,
              \mathbb{P}\big(\textstyle\cap_{r\in S} W_r \,\big|\, z\big)
$$

$S$ varie sur tous les ensembles de localisation downstream que l'on peut créer (quand on a que deux downstream on revient à la formule classique de $\mathbb{P}(A\cup B) = \mathbb{P}(A)+\mathbb{P}(B) - \mathbb{P}(A\cap B)$). Dans le cas simple où il n'y a qu'une entreprise downstream, disons localisée en $r$, on a directement: 

$$\tilde{\rho}_{r's}(z) = \rho_{r'rs}(z) = e^{-(T_{r's} \gamma_{r'rs}^{-1}z^{-\theta})} $$

Dans ce cas on aurait $$\log (- \log \tilde{\rho}_{r's}(z)) =   \theta\log \tau_{r'r}-\theta \log z + \log \Phi_{rs}  $$
Si jamais on paramétrise $\tau_{r'r} = d_{r'r}^\alpha$ alors on a : 

$$
\log (-\log \tilde{\rho}_{r's}(z)) = \alpha\theta \log d_{r'r} - \theta \log z + \log \Phi_{rs}
$$

On aurait envie de tester cette équation dans les données. Je ne sais pas que modèle permet de tester cette équation. Empiriquement, soit $i$ une entreprise de $(r',s)$, on teste :

$$Sup_{i} = \mu_{sr^*(i)} + \beta \log d_{r^*(i)}  + \nu \log X_{i} + \epsilon_{_i}$$

Où $Sup_{i}$ est la probabilité que l'entreprise fournisse à l'industrie downstream, $d_{r^*(i)}$ la distance à l'établissement downstream le plus proche, $\mu_{sr^*(i)}$ un effet fixe Secteur $\times$ Etablissement downstream le plus proche et $X_i$ un proxy pout la taille de l'entreprise. Si jamais on fait un conditional logit, ça revient à faire la régression suivante $$\log \frac{\tilde{\rho}_{r's}}{1-\tilde{\rho}_{r's}(z)}(z) = \mu_{sr^*(i)} + \beta \log d_{r^*(i)}  + \nu \log X_{i} + \epsilon_{_i}$$**Conséquence**

J'utilise maintenant le $\beta$ de cette régression comme initialiseur de $\alpha$. 

# Pourquoi les avantages comparatifs sont-ils mal estimés ?

L'estimateur de la variance est l'estimateur Sandwich. Le coeur de cet estimateur est la viande du sandwich c'est à dire le terme $J W J^T$ où $J$ est la jacobienne de notre système et $W = \Omega^{-1}$ avec $\Omega$ la matrice de variance covariance du système.  En général, les erreurs standard explosent lorsque : 

1) Le système n'est pas identifiable. 
2) La jacobienne n'est pas bien estimée. 
3) L'algorithme ne converge pas vers un point d'équilibre stable.

Dans le reste de la discussion, on s'intéresse à la partie de cette matrice qui concerne que $\alpha$ et $T$. 

**1) Système non identifiable : Est-il possible d'estimer les avantages comparatifs ?**

Pour estimer les avantages comparatifs, il faut que le noyau de la Jacobienne soit nul. Si jamais la dimension du noyau est plus grande que $M-P$ (# Moments - # Paramètres) alors on ne peut pas inverser le système. Pour les avantages comparatif, le noyau est restreint au vecteur constant. Une fois que je fixe une région de référence alors la projection est injective. 

_Point de vigilance :_ On pourrait penser qu'il existe un second vecteur propre du noyau. Prenons trois régions {1,2,3}. La première est la région de référence. Augmenter $T_2$ semble être équivalent à diminuer $T_3$. Mais comme on a fixé la valeur de $T_1$ alors ça ne l'est pas. Une fois la région de référence fixée, augmenter le T d'une région ne peut être compensé en augmentant le T d'une autre région. 


**2) Est ce que l'estimation de la Jacobienne est bonne** ? 

Si la Jacobienne est bruitée alors elle va détériorer l'estimateur Sandwich. Voici une raison pour laquelle elle est bruitée. 

 Soit $\omega$ un bruit, $\Theta$ le vecteur de paramètres et $m(\Theta,\omega)$ le vecteur de moments calculé à partir de l'aléa $\omega$. On a : 

$$J_{ij}(\omega) = \frac{m(\Theta_i^+,\omega)_j-m(\Theta_i^-,\omega)_j}{2h_i} $$

où $\Theta_i^+ = (\Theta_1,...,\Theta_i + h_i, .... \Theta_N)$ le vecteur de paramètre que l'on perturbe à la position _i_. On répète $K = 50$ fois cette opération avec des $\omega$ différents et on obtient la jacobienne en moyennant sur l'échantillon. Les paramètres ont des échelles différentes on définit $h$ proportionnellement à la valeur du paramètre : $h_i = \Theta_i \times \kappa$. 

Etant donné que l'on approxime la jacobienne par une moyenne sur un estimateur par différence première, il est possible que le bruit de simulation $\omega$ ne se moyenne pas et que la Jacobienne reste bruitée si elle présente des discontinuités.

Pour voir pourquoi le bruit ne se moyenne pas, plaçons-nous sur une seule entreprise de productivité $z$ tirée à partir de l'aléa $\omega$, de densité $f$. Le seuil $z^*(\alpha)$ est supposé lisse et strictement monotone en $\alpha$. La contribution de cette entreprise à la différence première s'écrit :

$$D(\omega) = \frac{\mathbb{1}\{z < z^*(\alpha + h)\} - \mathbb{1}\{z < z^*(\alpha - h)\}}{2h}$$

Cette quantité est **nulle partout**, sauf lorsque $z$ tombe dans la fine bande située entre les deux seuils, $$ z \in \big[,z^*(\alpha - h),\ z^*(\alpha + h),\big], \qquad \text{de largeur} \quad \Delta \approx 2h,\lvert z^{*\prime}(\alpha)\rvert, $$ et dans cette bande elle vaut $\tfrac{1}{2h}$. Autrement dit, $D(\omega)$ ne prend que deux valeurs : $0$ ou $\tfrac{1}{2h}$.

**L'estimateur est bien centré.** En moyenne, la probabilité de tomber dans la bande est $p \approx f(z^*)\Delta = 2hf(z^*)\lvert z^{*\prime}\rvert$, donc

$$ \mathbb{E}[D] = \frac{1}{2h},p \approx f(z^*),\lvert z^{\prime}(\alpha)\rvert, $$

qui est bien la vraie dérivée. Le problème n'est donc pas un biais, mais la **variance**.

**La variance explose quand $h \to 0$.** Comme $D$ est $\tfrac{1}{2h}$ fois une Bernoulli de paramètre $p$,

$$ \operatorname{Var}(D) = \frac{1}{4h^2}p(1-p) \approx \frac{1}{4h^2}\cdot 2hf(z^*)\lvert z^{\prime}\rvert = \frac{f(z^*)\lvert z^{\prime}\rvert}{2h} \xrightarrow[h\to 0]{} \infty. $$

En moyennant sur $K$ tirages de $\omega$, la variance de l'estimateur de la Jacobienne vaut $\operatorname{Var}(D)/K \approx \dfrac{f(z^*)\lvert z^{*\prime}\rvert}{2hK}$. Pour la contrôler, il faudrait $K \gg 1/h$ : avec $h$ petit et $K = 50$ fixé, le bruit de simulation **ne se moyenne pas**. On peut atténuer ce bruit en augmentant $h$, au prix d'un biais de troncature en $O(h^2)$ (développement de Taylor). Ou sinon on peut remplacer la Jacobienne empirique par une version analytique mais on n'a pas de formule différentiable (cf prochaine discussion).

**Contraste avec un moment lisse.** Si $m(\Theta,\omega)$ était différentiable en $\Theta$, la différence première aurait une variance bornée (de l'ordre du bruit de $m$ lui-même) et un biais en $O(h^2)$ : moyenner sur $K$ tirages suffirait alors à la stabiliser. C'est précisément la discontinuité de l'indicatrice — le passage brutal de fournisseur à non-fournisseur — qui fait diverger le rapport signal/bruit en $1/h$ et laisse la colonne correspondante de la Jacobienne bruitée, ce qui peut se propager en une petite valeur propre de $G'WG$.


**Solution**: Dans sont papier de multi-stage production Thierry fait ça : 

_Confidence intervals given by the second lowest and second highest of 40 bootstrap samples drawn with replacement. Sampling within headquarters continents ensures that the bootstrap sample moments include activity on each continent_ 

Il a 40 versions de ses moments (perso je trouve ça faible étant donné qu'il a 40 moments donc l'estimation d'une matrice de variance covariance est très mauvaise) et il fait l'optimisation à chaque fois. Ensuite il calcule les intervalles de confiance sur l'ensemble des paramètres qu'il obtient. 

Ca évite de devoir estimer la Jacobienne mais je n'aime pas car : 

i) Ca ne traite pas le bruit de simulation. 
	On pourrait indépendamment calculer la matrice de covariance des données empirique par bootstrap, celle des données simulées et comparer l'ampleur du bruit des données simulées. 
ii) Ca suppose que l'algorithme d'optimisation n'est pas coincé dans un minimum local. 


**3) Est-ce que la Jacobienne est informative ?**

Même si le modèle est bien spécifié (il est injectif), il est possible que certaines directions du modèle ne soit pas informatives (espace propre avec des directions très proche du noyau / plates) ce qui a pour effet d'élargir les intervalles de confiance. C'est le cas notamment quand la jacobienne est très proche de $0$ sur certaines entrées. 

Pour le montrer, concentrons nous sur la Jacobienne des $\gamma$ par rapport à $T$. Tout d'abord cette Jacobienne est diagonale par block si jamais les effets entre les secteurs sont faibles. Ensuite on a :

$$
J_{r'k} = \frac{1}{T_k} \sum_r w_r \gamma_{r'r}(\mathbf{1}\{r'=k\} - \gamma_{r'r})
$$

où $\gamma_{r'r}$ est la part du sourcing de $r$ fait auprès de $r'$. La jacobienne est très faible si les parts sont proches de 0 ou de 1 (peu identifiante). 

L'estimateur de Sandwich est $Var(\Theta) = (J'WJ)^-1 J'W \Omega W J (J'WJ)^{-1}$ où $W$ est la matrice de pondération de l'estimation (l'identité dans le premier passage) et $\Omega$ la matrice de variance-covariance du modèle qui encode le bruit de simulation et le bruit des données ($\Omega = \Omega_{data} + \Omega_{simulation}$).

Dans la première passe, on suppose que $W = \Omega = I$ alors $Var(\Theta) = ((J'J)^{-1})^2 = Q Diag(\lambda^2)^{-1} Q'$. Avec $Diag(\lambda)$ la matrice des valeurs propres de $J'J$.  On peut montrer que $$\max_{||u|| = 1} Var(u\Theta) = \frac{1}{\lambda_{min}}$$. Si $\lambda_{min}$ est petit alors il existe une direction dans l'espace des paramètres le long de laquelle l'optimisation distingue très mal les paramètres. De plus on peut montrer que $\lambda_{min} < \min_r \gamma_{r}$ donc si jamais on a des faibles quantité alors on va avoir une forte variance. 

Le bon ratio à regarder est le "condition number", le rapport entre les valeurs propres. J'observe qu'il est raisonnable dans le cas de ma matrice. 


**4) L'algorithme ne converge pas vers un point d'équilibre stable**

Toute la discussion précédente suppose que l'on est bien au vecteur optimal $\Theta^*$. Mais il se peut que l'on n'y soit pas. 

L'algorithme d'optimisation est Particule Swarm Algorithm. C'est un algorithme qui a l'avantage de ne pas avoir besoin de différencier la loss function mais qui a l'inconvénient qu'il faille définir des bornes pour nos paramètres. Aujourd'hui, je pars d'un point initial pour le vecteur $(\alpha,\mathbf{T})$  et j'explore l'ensemble des points qui sont compris dans un rayon de $\pm 20\%$ autour de ce point (voir $50$). Le problème avec ce type d'algorithme c'est qu'il est très sensible à la valeur initiale et donc, nous ne sommes pas certains de tomber sur le minimum global du problème. 

Ce que j'ai envie de faire c'est d'utiliser un algorithme de descente de gradient mais pour ça il faut pouvoir différencier tous les moments par rapport aux paramètres. Le seul moment qui pose problème c'est la régression à la marge extensive dont on a une formule théorique mais difficile à calculer. 

$$
\tilde{\rho}_{r's}(z) = \mathbb{P}(\cup_r W_r | z ) = \sum_{\emptyset\neq S\subseteq\mathcal{D}} (-1)^{|S|+1}\,
              \mathbb{P}\big(\textstyle\cap_{r\in S} W_r \,\big|\, z\big)
$$


On pourrait réduire le nombre de downstream d'intérêt (par exemple en prenant les X plus grandes ou les X plus proches). 


**Estimer les T par méthode du point fixe**

Actuellement comme on n'a pas de prior concernant la distribution des avantages comparatifs. Néanmoins, conditionnellement à $\alpha$, on peut trouver les avantages comparatifs. Pour exposer la méthode, je propose de supprimer l'indice sectoriel. On a : 

$$\gamma_{r'} = \sum_{r} w_r \frac{T_{r'}d_{r'r}^{-\alpha \theta}}{\sum_{r'} T_{r'}d_{r'r}^{-\alpha \theta}}$$
On peut donc poser $d_{r'r}^{-\alpha \theta} = K_{r'r}$, $(KT)_r = \sum_{r'}T_{r'}d_{r'r}^{-\alpha \theta}$ et $s_r = \frac{w_r}{(KT)_r}$. Ainsi, 

$$\gamma_{r'} = T_r' \sum_{r} K_{r'r} s_r$$ On introduit la matrice $F$ tel que $$F_{r'r} = T_r' K_{r'r}s_r$$ Sommer les lignes de cette matrice donne $w_r$ et sommer les colonnes donne $\gamma_{r'}$ (**marges**). Si jamais les $\gamma$ sont les parts domestiques, alors on a $\sum_r w_r = \sum_{r'} \gamma_{r'} = 1$.


----

**_Théorème (Sinkhron 1967):_** Soit ($w,\gamma$) tel que $\sum_r w_r = \sum_{r'} \gamma_{r'}$ et $K_{r'r} >0 \ \forall (r',r)$, alors il existe des vecteurs positifs `T`, `s` tels que  $\operatorname{diag}(T)\,K\,\operatorname{diag}(s)$ ait ces marges, **uniques à l'échelle près**  $(T,s)\mapsto(cT, s/c)$.

-----

Ainsi, en fixant une région de référence, on fixe l'échelle et donc il existe un $T$ et $s$ unique. 

L'algorithme associé au théorème est celui de **Sinkhorn–Knopp** : il consiste à mettre à jour alternativement $T$ et $s$ pour imposer, tour à tour, chacune des deux marges. Comme $F_{r'r}=T_{r'}K_{r'r}s_r$, imposer la marge en ligne $\gamma$ fixe $T$ à $s$ donné, et imposer la marge en colonne $w$ fixe $s$ à $T$ donné :

$$ T_{r'} ;\leftarrow; \frac{\gamma_{r'}}{\sum_r K_{r'r},s_r}, \qquad s_{r} ;\leftarrow; \frac{w_{r}}{\sum_{r'} K_{r'r},T_{r'}}. $$

On itère jusqu'à ce que les deux marges soient satisfaites, en renormalisant à chaque passe par la région de référence pour fixer l'échelle libre $(T,s)\mapsto(cT,s/c)$.

**Algorithme (Sinkhorn–Knopp)**

```
Entrée : matrice K  (K_{r'r} = d_{r'r}^{-αθ}),
         cibles γ (marge-ligne), w (marge-colonne),
         région de référence r0, tolérance ε
Initialiser  s_r ← 1  pour tout r

répéter jusqu'à convergence :
    # (1) imposer la marge-ligne γ  →  met à jour T
    pour chaque r' :   T_{r'} ← γ_{r'} / Σ_r  K_{r'r} s_r

    # (2) imposer la marge-colonne w  →  met à jour s
    pour chaque r  :   s_r  ← w_r  / Σ_{r'} K_{r'r} T_{r'}

    # (3) fixer l'échelle sur la région de référence
    c ← T_{r0} ;  T ← T / c ;  s ← s · c

    # critère d'arrêt : violation maximale des marges
    δ ← max( max_{r'} |Σ_r F_{r'r} − γ_{r'}| , max_r |Σ_{r'} F_{r'r} − w_r| )
jusqu'à  δ < ε

Retourner T  (et s)
```

À la convergence, $T$ contient les avantages comparatifs conditionnels à $\alpha$, normalisés à la région de référence. (En pratique on effectue les deux mises à jour en espace-log avec un amortissement $\lambda \approx 0{,}5$, $\log T \leftarrow (1-\lambda)\log T + \lambda\log \hat T$, ce qui évite les oscillations sans changer le point fixe.)


On a donc un algorithme qui permet, conditionnellement à $\alpha$ de trouver le vecteur $T$ optimal. On peut donc partir d'un point initial pour $T$ autour duquel on fait le PSO. 

**5) Extension**

On peut alléger le PSO en ne cherchant plus $T$ : à chaque particule, on propose un $\alpha$ (et les paramètres de tête $\Omega, A$), et on **récupère $T$ par Sinkhorn–Knopp** conditionnellement à $\alpha$. La subtilité est que la marge-colonne $w_r$ (dépenses/parts régionales) est un objet d'**équilibre** : quand $T$ bouge, les prix et les revenus bougent, donc $w$ bouge aussi. La cible même de l'algorithme de Sinkhorn se déplace. Il faut donc emboîter Sinkhorn dans une boucle de point fixe sur $w$ :

```
solve_T_given_alpha(α, γ_data ; réf r0) :
    w ← w0                       # init : parts de dépense observées
    répéter (boucle d'équilibre général) :
        K ← [ d_{r'r}^{-αθ} ]
        # (a) T conditionnel à (α, w, γ) — point fixe interne de Sinkhorn
        T ← Sinkhorn(K ; marge-ligne = γ_data, marge-colonne = w, réf = r0)
        # (b) équilibre : ce T induit de nouveaux prix / revenus / dépenses
        (prix, revenus) ← résoudre_équilibre(T, α)
        w_new ← parts de dépense régionales impliquées
        δ ← || w_new − w ||
        w ← relax · w_new + (1−relax) · w      # sous-relaxation, stabilité
    jusqu'à  δ < ε_GE
    retourner (T, w)

PSO sur (α, Ω, A) :
    pour chaque particule proposant (α, Ω, A) :
        (T, w) ← solve_T_given_alpha(α, γ_data)
        m ← moments_modèle(α, Ω, A, T, w)
        L ← (m − m_data)' W (m − m_data)   # sur les moments RESTANTS
    mettre à jour les particules
retourner (α̂, Ω̂, Â) et le T̂ associé
```

Deux boucles emboîtées : la boucle interne (Sinkhorn) résout $T$ à $w$ fixé, la boucle externe (équilibre général) met à jour $w$ jusqu'au point fixe. Un point important : comme $\gamma$ est **exactement reproduit** par construction (c'est la marge imposée à Sinkhorn), le bloc $\gamma$ sort de la fonction objectif — le PSO n'ajuste plus que $\alpha$ et les paramètres de tête sur les moments restants. On remplace ainsi une recherche en grande dimension (tous les $T_{r's}$) par une résolution exacte, et le PSO ne balaie plus qu'un espace de faible dimension.

L'intervalle de confiance est construit de la même manière que précédemment : on calcule l'estimateur sandwich à partir de la Jacobienne simulée au point optimal et on utilise la matrice identité comme matrice de poids. 


**6) Variance de $T$**

Aujourd'hui les $T$ ont des IC qui couvrent 0. Deux commentaires : 

1) Etant donné que $T > 0$, est-ce que l'on doit utiliser un autre estimateur de la variance ? 
2) Ce qu'on mesure c'est plutôt le ratio $T/T_{ref}$. Est-ce qu'on doit plutôt tester la différence entre $T$ et $T_{ref}$ (test de Wald à 1) ?



# Questions 

- [ ] Est-ce que l'on ne doit calculer les coefficients de la régression à la marge intensive en utilisant la méthode cloglog ? 
	- [ ] Suggestion: Dans le papier on continue de présenter le LPM dans le main texte et on fait la même régression en utilisant le modèle cloglog pour la partie SMM.
- [ ] Est-ce qu'on passe sur le Sinkhorn dans la boucle ? Ca permet de supprimer $T$ de l'estimation mais par contre on a sûrement envie d'avoir plus de moments pour identifier $\alpha$ car toute l'estimation de $T$ va reposer sur a quel point on estime bien $\alpha$. (Pour l'automobile on passe de 166 paramètres à 33 paramètres quand même...)
- [ ] Estimer les intervalles de confiance en faisant tourner l'optimisation sur des bootstrap indépendants X fois. 
- [ ] Passer en TikTak qui peut être plus robuste. 