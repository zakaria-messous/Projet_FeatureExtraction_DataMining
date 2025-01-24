# Projet de Sélection de Caractéristiques et Évaluation de Classification

## Aperçu
Ce projet explore l'application des méthodes de sélection de caractéristiques :
1. Score de Fisher
2. Chi-Carré (Chi-Square)
3. Information Mutuelle
4. Forward Feature Selection

Chaque méthode de sélection de caractéristiques est suivie par l'évaluation de trois algorithmes de classification :
- Random Forest
- Support Vector Machine (SVM)
- Multinomial Naive Bayes

L'objectif est d'évaluer la performance de chaque combinaison et de déterminer l'approche la plus efficace pour prédire la performance des étudiants.

## Structure du Projet
Le projet est organisé comme suit :

```plaintext
Répertoire du Projet
|
|-- feature_chi2/                # Scripts et résultats pour la sélection de caractéristiques avec Chi-Carré
|-- FishersScore/                # Scripts et résultats pour la sélection de caractéristiques avec le Score de Fisher
|-- forward_feature_selection/   # Scripts pour la sélection de caractéristiques avant
|-- Information_mutuelle/        # Scripts et résultats pour la sélection de caractéristiques avec Information Mutuelle
|-- TF-IDF/                      # Contient la matrice TF-IDF pré-traitée
|-- DataSet/                     # Jeu de données original utilisé pour le projet
|-- README.md                    # Ce fichier
```

## Jeu de Données
Le jeu de données utilisé pour ce projet contient des informations sur les performances académiques des étudiants et provient de [Kaggle](https://www.kaggle.com). Vous pouvez télécharger le jeu de données directement à partir du lien suivant :

[Student Performance Dataset - Kaggle](https://www.kaggle.com/datasets/devansodariya/student-performance-data)

## Flux de Travail

### 1. Pré-traitement des Données
- Le jeu de données original est pré-traité pour nettoyer les valeurs manquantes et normaliser les données.
- La variable cible, `G3` (note finale), est transformée en trois catégories :
  - **Low (0-10)**
  - **Medium (11-15)**
  - **High (16-20)**
- Les données sont converties en une matrice TF-IDF pour représenter numériquement les caractéristiques textuelles.

### 2. Méthodes de Sélection de Caractéristiques
Trois techniques de sélection de caractéristiques sont appliquées pour identifier les caractéristiques les plus pertinentes :

#### Score de Fisher
- Mesure la capacité d'une caractéristique à discriminer entre les catégories.
- Produit les 10 meilleures caractéristiques en fonction de leurs scores.

#### Chi-Carré
- Évalue l'indépendance entre les caractéristiques et la variable cible.

#### Information Mutuelle
- Mesure la quantité d'information qu'une caractéristique apporte pour prédire la variable cible.

### 3. Algorithmes de Classification
Chaque ensemble de caractéristiques sélectionné est testé avec trois algorithmes de classification :

- **Random Forest** : Modèle basé sur des arbres de décision multiples, reconnu pour sa robustesse.
- **Support Vector Machine (SVM)** : Modèle qui sépare les classes à l'aide d'hyperplans.
- **Multinomial Naive Bayes** : Classificateur probabiliste efficace pour les données catégoriques.

### 4. Métriques d'Évaluation
La performance de chaque modèle est évaluée en utilisant :
- **Précision** : Proportion d'exemples correctement classés.
- **Précision, Rappel, F1-Score** : Métriques pour chaque classe (Low, Medium, High).
- Des visualisations des métriques de performance sont générées pour comparaison.

## Résultats
Chaque combinaison de méthode de sélection de caractéristiques et de classificateur est évaluée. Les résultats sont enregistrés dans les répertoires respectifs sous `FishersScore`, `feature_chi2`, et `Information_mutuelle`. Des graphiques comparatifs et des tableaux sont également fournis.

### Principaux Constats
- **Score de Fisher + Random Forest** a obtenu la meilleure précision globale.
- **Information Mutuelle + SVM** a montré des performances équilibrées entre toutes les classes.
- Les métriques détaillées sont disponibles dans les répertoires de résultats respectifs.

## Comment Exécuter le Projet
1. Clonez le répertoire :
   ```bash
   git clone <repository_link>
   ```
2. Installez les packages Python requis :
   ```bash
   pip install -r requirements.txt
   ```
3. Exécutez les scripts pour chaque méthode de sélection de caractéristiques et classificateur. Par exemple :
   ```bash
   python FishersScore/run_fishers_score.py
   python feature_chi2/run_chi2.py
   python Information_mutuelle/run_mutual_info.py
   ```
4. Les résultats seront enregistrés dans les répertoires respectifs.

## Citation du Jeu de Données
Veuillez citer la source originale du jeu de données si vous utilisez ce projet à des fins de recherche ou académiques :

- **Kaggle - Student Performance Dataset**

## Travaux Futurs
- Étendre l'analyse pour inclure des modèles plus avancés comme XGBoost et LightGBM.
- Appliquer des techniques supplémentaires de sélection de caractéristiques comme l'élimination récursive des caractéristiques (RFE).
- Expérimenter avec des modèles d'apprentissage profond pour des prédictions plus précises.

---

Pour toute question ou contribution, n'hésitez pas à nous contacter !

