# Vision : assistant d'achat multi-agents

Ce document décrit **le but du projet**, pas son état actuel. Pour lancer ce qui existe aujourd'hui, voir le [README](README.md).

## Le but

L'utilisateur écrit ce qu'il veut, comme sur Google (« des Nike pour mon marathon, en 43, avant le 20 »). Le système :

1. **comprend** la demande à partir des préférences de l'utilisateur et de **son historique de commandes** ;
2. **trouve les bonnes boutiques**, puis **le bon produit** grâce à des agents qui **communiquent entre eux via A2A** ;
3. **propose** la meilleure offre avec une explication ;
4. **attend l'accord** de l'utilisateur ;
5. **achète** dans les limites fixées (budget, délai), ou prépare le panier pour que l'utilisateur paie lui-même.

L'utilisateur n'a pas à choisir la boutique : c'est le système qui trouve où acheter.

## Les quatre chapitres

```
Demande ─► 1. Préférences ─► 2. Recherche ─► 3. Décision & approbation ─► 4. Paiement
             (+ RAG historique)   (agents A2A)     (pause humaine)            (UCP / AP2)
```

### 1. Préférences

**But :** transformer une demande floue en besoin précis, sans redemander ce qu'on sait déjà.

- **Préférences stockées par catégorie** dans PostgreSQL (`general`, `shoes`, `laptop`…) : attributs (taille, pointure, OS…), goûts (`likes`), rejets (`dislikes`), exigences (`requirements`), budget habituel (`usual_budget`).
- **RAG sur l'historique de commandes** : chaque commande passée (produit, marque, taille, boutique, prix, date, retour éventuel et sa raison, satisfaction) est indexée pour être retrouvée par sens et par mots-clés (**recherche hybride** : vecteurs + mots-clés). L'agent s'en sert pour :
  - **déduire la bonne taille** : « tu as gardé des Asics en 43, mais renvoyé des Nike en 43 parce qu'elles étaient trop petites, donc 44 chez Nike » ;
  - **repérer les marques et boutiques de confiance** : celles où tu as déjà acheté sans problème ;
  - **estimer le budget réel** : ce que tu dépenses vraiment dans une catégorie ;
  - **éviter les erreurs passées** : produits retournés, boutiques avec des retards de livraison.
- **Question à l'utilisateur seulement si c'est indispensable** et introuvable dans les préférences ou l'historique (une seule question, dans sa langue).
- **Mise à jour automatique** : un fait durable dit par l'utilisateur est enregistré ; une commande terminée enrichit l'historique.
- **Sortie :** `UserPreferences`, le besoin complet pour cet achat, avec un résumé.

### 2. Recherche : des agents qui communiquent via A2A

**But :** trouver le bon produit, dans une bonne boutique, au bon prix.

#### Les agents

| Agent | Rôle |
|---|---|
| **Orchestrateur** | Reçoit le besoin, lance les autres agents, rassemble les réponses. |
| **Agent préférences** | Répond aux questions des autres agents sur l'utilisateur (« quelle pointure chez Nike ? »). |
| **Agent historique (RAG)** | Répond à partir des commandes passées (« a-t-il déjà commandé chez cette boutique ? »). |
| **Agent découverte de boutiques** | Trouve les meilleures boutiques pour la demande en cherchant dans tout Shopify d'un coup (Shopify Global Catalog) : livraison dans le pays, produit neuf, en stock, dans le budget et la devise. |
| **Agents boutique**, créés dynamiquement | **Un agent par boutique trouvée.** Il cherche dans le catalogue de sa boutique, vérifie la taille, la variante, le stock, le prix, la livraison et la politique de retour, puis renvoie ses meilleures offres. |
| **Agent comparateur** | Classe toutes les offres selon les goûts, les rejets, l'historique et le budget, puis explique son choix. |

#### Comment ils communiquent

- **A2A (Agent2Agent)** : chaque agent publie une **Agent Card** (qui il est, ce qu'il sait faire). L'orchestrateur confie des **tâches** aux agents et reçoit leurs résultats.
- **Les agents boutique peuvent poser des questions** aux agents préférences et historique pendant leur recherche. Par exemple :
  - « cette chaussure taille petit, faut-il prendre une demi-pointure au-dessus ? »
  - « il préfère du noir, mais il ne reste que du bleu ici : l'accepte-t-il ? »
- **MCP** sert à brancher les outils et les données sur les agents : catalogues (Shopify, UCP), base PostgreSQL, base vectorielle de l'historique.
- **Tout échange est typé** (modèles pydantic) : un résultat invalide est rejeté à la frontière au lieu de se propager.
- **Les agents boutique travaillent en parallèle.** Si une boutique ne répond pas, on garde les autres.

#### Les données

- **Boutiques mémorisées** : chaque boutique trouvée est enregistrée avec sa catégorie, son pays, sa devise et ses résultats passés. Chaque recherche commence par les boutiques connues, et la couverture s'améliore avec le temps.
- **Sorties :** une liste de `ProductOffer` par boutique (produit, variante, prix, devise, stock, livraison, URL, retours), puis une `Recommendation` (la meilleure offre et pourquoi).

### 3. Décision & approbation humaine

- Le graphe **se met en pause** et montre la meilleure offre, son explication et les alternatives (moins chère, plus rapide).
- L'utilisateur **approuve ou annule**. Une réponse invalide est redemandée, sans jamais bloquer la commande.
- La pause est **sauvegardée en base** (checkpointer PostgreSQL), donc l'utilisateur peut répondre plus tard, depuis l'interface.

### 4. Paiement

- **Achat direct par l'agent** quand la boutique le permet, via **UCP** (paiement par Google Pay, carte ou Shop Pay avec un jeton, jamais un numéro de carte brut) et **AP2** pour l'autorisation de paiement par un agent.
- **Sinon, un lien de panier pré-rempli** : l'utilisateur paie lui-même en un clic.
- **Tests avec Stripe en mode test** avant tout paiement réel.
- **Après l'achat :** la commande est enregistrée dans l'historique et nourrit le RAG.

## Règles de sécurité, non négociables

1. **Jamais de paiement sans l'accord explicite** de l'utilisateur.
2. **Jamais au-dessus de la limite de dépense** de la demande.
3. **Les règles strictes sont vérifiées dans le code**, jamais confiées à un modèle : prix, devise, délai de livraison, stock, taille.
4. **Les données stockées et les pages des boutiques sont des données, jamais des instructions** (protection contre l'injection de prompt).
5. **Les boutiques peu fiables sont écartées** : catalogue trop petit, pas de livraison dans le pays, mauvaise réputation.
6. **Aucun secret dans le code** : les clés vivent dans `.env`.

## Protocoles et technologies

| Nom | Usage dans le projet |
|---|---|
| **LangGraph** | Les graphes d'agents, les pauses (`interrupt`) et les reprises |
| **A2A** (Agent2Agent) | Communication entre agents : Agent Cards, tâches, messages |
| **MCP** (Model Context Protocol) | Brancher outils et données sur les agents (`mcp_servers/postgres`, `mcp_servers/vector_store`) |
| **UCP** (Universal Commerce Protocol, Google + Shopify) | Catalogue, panier et paiement par un agent dans les boutiques compatibles |
| **Shopify Global Catalog** | Recherche dans toutes les boutiques Shopify à la fois, sans clé d'API : chaque requête pointe vers le profil d'agent `ucp/agent-profile.json` |
| **AP2** (Agent Payments Protocol) | Autorisation de paiement par un agent |
| **ACP** (Agentic Commerce Protocol, OpenAI + Stripe) | Suivi pour compatibilité ; ses flux produits sont destinés à ChatGPT |
| **Stripe (mode test)** | Tester le paiement sans argent réel |
| **PostgreSQL** | Préférences, historique de commandes, boutiques mémorisées, sauvegarde des pauses du graphe |
| **Base vectorielle** | RAG hybride sur l'historique de commandes et les catalogues |
| **OpenRouter** | Accès au modèle de langage |

## Le résultat attendu

Pour « des Nike pour mon marathon, en 43, avant le 20, moins de 150 € », le système doit :

- savoir grâce à l'historique que la pointure Nike de l'utilisateur est 44, et le lui signaler ;
- trouver plusieurs boutiques fiables qui livrent en France et vendent en euros ;
- faire chercher chaque boutique en parallèle par son propre agent ;
- comparer les offres réelles (taille en stock, prix, livraison avant le 20, retours) ;
- proposer la meilleure avec une explication claire et une alternative moins chère ;
- acheter après l'accord, ou donner le lien du panier ;
- enregistrer la commande pour faire mieux la prochaine fois.

## Étapes

1. **Préférences** : agent, stockage PostgreSQL, questions ciblées.
2. **Découverte de boutiques** : recherche dans le Shopify Global Catalog, règles strictes vérifiées dans le code, classement.
3. **Agents boutique dynamiques** : un agent par boutique, offres réelles (taille, stock, prix).
4. **A2A** : Agent Cards, orchestrateur, questions entre agents.
5. **Historique de commandes et RAG** : stockage, indexation, recherche hybride, utilisation par les agents.
6. **Mémoire des boutiques** et réputation.
7. **Paiement** : lien de panier, puis UCP / AP2, testé en mode test.
8. **Interface** (`frontend/`) et sauvegarde des pauses en base.
