1.	Fase 1 - Probing linguistico
Studio delle rappresentazioni interne di BERT
1.	Ridge Regressor + MinMaxScaler su feature UD (8000train / 2000 test)
2.	Rappresentazione frase via token [CLS] - Probing su tutte le features (63) dell'ultimo layer (per iniziare)
3.	Ref: vedi paper sotto ACL 2020 (Contextual Embeddings - Linguistic Profiling)
2.	Fase 2 - Studio teorico: DIffusion & Flow Matching
Comprendere il pretraining dei modelli generativi per il linguaggio
1.	Come si addestrano modelli basati su Diffusion (HuggingFace)
2.	Studio di Flow Matching come alternativa (se possibile nei tempi)
2.	Fase 3 - Preparazione dataset e ambiente
Dataset fornito (Wikipedia): frasi singole o blocchi di frasi
1.	Filtrare frasi con len(frase) <6 token (poco informative)
2.	Configurare ambiente HuggingFace - Modello base per le prime run equivalente a BERT 2x2
2.	Fase 4 - Nuova idea: Diffusion + Bidirectional Mamba
Sostituire la backbone Transformer di un Diffusion Model (HuggingFace) con Bi-Mamba
1.	Capire come addestrare un modello Bi-Mamba da zero
2.	Partire da modello molto piccolo (~BERT 2x2)
3.	Salvare checkpoint a granularità variabile (frequenti all'inizio → meno frequenti alla fine)
2.	Fase 5 - Analisi comparativa e probing sul pretraining
Confronto tra Bi-Mamba e Transformer (modello addestrato da Luca, 40M di parametri)
1.	Probing linguistici a diversi checkpoint durante il pretraining
2.	Confronto Bi-Mamba vs Transformer allo stesso punto di addestramento
3.	Analisi: quale backbone apprende prima le strutture linguistiche? 
Obiettivo di tesi: valutare se Bidirectional Mamba implementato in un Diffusion Model apprende rappresentazioni linguistiche più rapidamente o meglio rispetto al Transformer nell'ambito dei modelli diffusivi → passo intermedio verso eventuale Flow Matching con Mamba backbone.
