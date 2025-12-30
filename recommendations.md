# Audit & Recommandations Stratégiques

Vous avez raison de me challenger. Jusqu'ici j'ai exécuté la mise en place technique, mais d'un point de vue **Data Science**, l'approche actuelle (Régression Linéaire sur le prix au jour le jour) a des limites structurelles majeures.

Voici mon analyse franche et mes recommandations pour rendre le projet réellement pertinent.

## 1. Le Constat d'Échec de la "Régression Simple"
Nous venons de voir que le modèle n'arrive pas à prédire les variations quotidiennes du prix officiel (`coefs` proches de 0).

*   **La cause** : Le "Prix Officiel" (Rosstat) est une donnée administrative, lissée, et contrôlée par l'État. Elle ne réagit pas au jour le jour aux rumeurs Telegram. Elle est "rigide".
*   **Le problème** : Essayer de prédire les ticks d'une courbe plate avec des données textuelles volatiles (Telegram) est voué à l'échec (bruit vs signal).

## 2. Ma Recommandation : Pivoter vers un "Système d'Alerte Avancée"

L'intérêt de Telegram n'est pas de deviner le prix de demain (inutile), mais de **détecter les crises avant qu'elles n'arrivent**.

### A. Changer la Target : De "Combien ?" à "Quand ?"
Au lieu de prédire la *valeur* du prix (`Regression`), nous devrions prédire le **Risque de Rupture** (`Classification`).
*   **Nouvelle Target** : `is_crisis` (Binaire : 0 ou 1).
*   **Définition** : Une période est "en crise" si la volatilité du prix dépasse le 90ème percentile OU s'il y a une hausse brutale (>2% en une semaine).

### B. Introduire le "Temps de Latence" (Lags)
C'est le point le plus critique. Sur Telegram, les routiers se plaignent **avant** que les stats officielles ne bougent.
*   **Action** : Créer des features décalées dans le temps.
*   `count_logistics_lag_7d` : Les plaintes d'il y a une semaine prédisent-elles le prix d'aujourd'hui ?
*   C'est ça qui donne de la valeur prédictive au modèle.

### C. Analyser la "Causalité" (Cross-Correlation)
Avant de lancer un modèle, faisons une analyse statistique simple :
*   *Quand la courbe "Logistique" monte, combien de jours plus tard le prix officiel réagit-il ?*
*   Cela nous donnera le "Lead Time" (ex: "Telegram donne 10 jours d'avance sur les stats officielles").

## 3. Analyse Post-Entraînement : Pourquoi la Logistique Gagne ?

Suite à l'entraînement du Classifieur Random Forest, nous avons la réponse à "Pourquoi `Logistics` fonctionne et pas les autres ?" :

### A. Cause vs Symptôme
*   **Logistique (Winner)** : Les problèmes de wagons/citernes ("RJD", "wagon") sont une **cause racine** opérationnelle. Ils surviennent *avant* que le carburant ne manque. C'est un **Indicateur Avancé**.
*   **Pénurie / Shortage (Loser)** : Les gens ne parlent de "pénurie" que quand la station est *déjà* vide. C'est un **Indicateur Retardé**. Au moment où Telegram crie "Pénurie", le prix a déjà monté d'Antigravity's side, donc ça n'aide pas à prédire le futur, c'est juste un constat du présent.

### B. Le cas du Sentiment (Null)
Pourquoi `sentiment_mean` est à 0 dans l'importance ?
*   **Raison Technique Honnête** : Pour accélérer les tests tout à l'heure, j'ai utilisé un script (`enrich_dataset.py`) qui a recalculé les compteurs, mais qui a rempli la colonne `sentiment` avec des zéros (car le NLP prend 3h à tourner).
*    *   Correction : Si on veut utiliser le sentiment, il faut impérativement lancer le calcul NLP complet cette nuit. Mais `fuel_stress_index` (qui est Top 4) a réussi à capturer une partie de l'info historique.

## 4. Et après ? Les Modèles à tester (Next Steps)

Nous avons atteint le "plafond de verre" du dataset Telegram seul (F1 ~ 0.04). Voici les pistes pour percer ce plafond :

### A. La Performance Pure : Gradient Boosting (XGBoost / CatBoost) [FAIT]
*   Validation : XGBoost optimisé donne des résultats similaires au Random Forest, confirmant que le frein est dans la donnée, pas le modèle.

### B. Ajout de Données Externes avec Prudence (Methodologie "Ablation")
*   **Le Danger** : Ajouter le Prix du Baril risque d'"éclipser" Telegram (le modèle ignorera le texte car le baril est trop prédictif).
*   **La Solution (USD/RUB)** : Le taux de change est un facteur de *stress* (pression à l'export) mais pas une cause directe mécanique. Il complète le signal Telegram sans le remplacer.
*   **Protocole Scientifique** : Faire une "Ablation Study".
    *   Modèle A : Telegram Uniquement (Notre baseline).
    *   Modèle B : Telegram + Macro (USD/RUB).
    *   *Succès* si Modèle B > Modèle A **ET** que les features Telegram restent dans le Top 10.

### C. Changer la Cible (Target Engineering)
*   **Problème** : Le prix *officiel* est "menteur" (contrôlé par l'État). Telegram parle du *vrai* marché.
*   **Solution** : Ne plus prédire le prix > 0.5%, mais prédire **"L'État de Crise"**.
    *   Créer une target composite : (Pénurie signalée OU hausse prix > 1% OU rationnement).
    *   Cela alignerait enfin la donnée (Telegram) avec l'objectif.

### D. Changer l'Échelle (Weekly vs Daily)
*   **Problème** : Prédire le jour J exact est trop dur (bruit).
*   **Solution** : Passer à une prédiction **Hebdomadaire**.
    *   Aggréger tout par semaine (`W-MON`).
    *   C'est beaucoup plus facile de dire "La semaine prochaine sera tendue" que "Mardi prochain le prix montera". Le score pourrait bondir à 60-70% de F1.

### E. L'Arme Fatale : Données Supply Chain (Interne & Externe)
*   **Interne (Nouveaux Canaux ajoutés)** :
    *   Nous avons ajouté les géants de l'info (`Mash`, `RIA`) et les spécialistes (`Neftegaz`, `Benzine`).
    *   *Stratégie* : Utiliser ces canaux pour le "Volume d'Alerte" (bruit de fond) tout en gardant les camionneurs pour le "Signal de Vérité".
*   **Externe** :
    *   Volumes d'exportation (Douanes).
    *   Planning de maintenance des raffineries (Indisponibilité offre).
    *   Prix du pétrole Urals (Rentabilité export).
