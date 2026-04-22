Pipeline X: Executive Summary & Visual Assets Guide
Questo documento descrive il funzionamento dell'architettura predittiva "Pipeline X" e fornisce una guida all'uso degli output visivi generati per le presentazioni agli stakeholder.

1. Architettura del Codice: Dall'Intelligenza all'Azione
La Pipeline X è progettata su una logica a "doppio binario" che separa l'estrazione delle previsioni (Machine Learning) dall'assegnazione dei prodotti (Business Logic e Compliance).

Fase A: I Modelli Predittivi Core (EBM e TabNet)
Il cuore predittivo del sistema è specializzato per target, utilizzando il modello migliore per ciascuno scopo:

EBM (Explainable Boosting Machine) per l'Accumulo (03x): Questo modello valuta la propensione del cliente verso prodotti di Accumulo. Utilizza la Tree View dei dati (valori grezzi e non scalati, come l'età reale o il patrimonio in euro). Questa scelta è fondamentale perché l'EBM è un modello "Glassbox": permette di generare regole trasparenti e leggibili dagli umani (es. "Patrimonio > 200.000€"), perfette per superare gli audit di conformità.

TabNet (Rete Neurale) per il Reddito/Income (04x): Per i prodotti a rendita, il sistema utilizza un'architettura neurale avanzata (TabNet). A differenza dell'EBM, TabNet riceve in input la NN View (dati scalati tra 0 e 1). Questa normalizzazione è obbligatoria per le reti neurali per evitare che variabili con magnitudo enormi (come la ricchezza) annullino l'importanza di variabili più piccole (come la propensione al rischio).

Fase B: Il Motore di Produzione e Compliance (05x)
I modelli predittivi indicano cosa vuole il cliente, ma non se è legale venderglielo. Lo script 05x_production_engine.py è l'arbitro finale che incrocia i bisogni predetti dall'AI con il catalogo prodotti della banca.
Applica tre filtri normativi (MIFID):

Coerenza: Il prodotto in catalogo deve corrispondere al bisogno predetto (Accumulo o Rendita).

Rischio: Il rischio del prodotto non deve mai superare la propensione al rischio dichiarata dal cliente.

Età: L'età del cliente deve rientrare nei limiti minimi e massimi previsti dal prodotto finanziario.

Il motore testa tre policy di severità aziendale: Strict (match perfetto), Age Gated (priorità all'età) e Closest (assegnazione del prodotto più simile per non perdere la vendita).

Fase C: L'Audit (06x)
Infine, il codice verifica che i modelli non discriminino i clienti (Fairness per età e genere) e valida la superiorità statistica della soluzione implementata rispetto alle baseline tradizionali.

2. Guida agli Asset Visivi per le Presentazioni
I seguenti grafici trasformano la complessità della Pipeline X in metriche di business comprensibili a qualsiasi manager o auditor.

🍩 Need Distribution Donut (05x_need_distribution_donut.png)
Cosa indica: Mostra come l'Intelligenza Artificiale ha segmentato la base clienti in base ai loro bisogni finanziari (es. percentuale di clienti propensi all'Accumulo vs Reddito).

Come usarla in presentazione: Questa è una slide di "Scenario Commerciale". Usala all'inizio per mostrare al team Marketing e Vendite la composizione della domanda latente. Risponde alla domanda: "Dove dovremmo concentrare le nostre campagne pubblicitarie questo mese?"

📊 Coverage Plot (05x_coverage_plot.png)
Cosa indica: Confronta tre metriche fondamentali: i clienti trovati dall'AI (potenziale), i clienti che hanno ricevuto un prodotto (vendite reali) e i clienti scartati dalle regole MIFID (vendite perse). Lo fa confrontando le tre policy (Strict, Age Gated, Closest).

Come usarla in presentazione: È la slide decisionale per il Board o il Product Manager. Usala per dimostrare che le mancate vendite non sono un errore dell'AI, ma dipendono da un catalogo prodotti inadeguato (es. mancano prodotti sicuri per anziani). Serve a far decidere all'azienda quanto rischio normativo vuole assumersi (scegliendo tra Strict e Closest) in cambio di maggiore fatturato.

🛡️ MIFID Safety Heatmap (05x_mifid_heatmap.jpg)
Cosa indica: Una matrice termica che incrocia la propensione al rischio del cliente (asse X) con il rischio del prodotto effettivamente assegnato dal sistema (asse Y).

Come usarla in presentazione: Questa è la slide per il team Legale e Compliance. Usala per fornire la prova visiva irrefutabile che il sistema è sicuro. Il punto di forza è far notare che il "triangolo del pericolo" (prodotti ad alto rischio assegnati a clienti prudenti) è completamente vuoto, garantendo zero violazioni MIFID.

🌊 Conversion Sankey Diagram (05x_conversion_sankey.html)
Cosa indica: Un diagramma di flusso interattivo che mappa l'intero viaggio dei clienti: dal totale iniziale, alla divisione nei vari bisogni predetti, fino all'esito finale (assegnazione del prodotto o scarto per motivi di compliance).

Come usarla in presentazione: È la "Hero Slide" conclusiva. Essendo interattiva, usala per fare storytelling: guida il pubblico con il mouse seguendo un flusso specifico (es. "Guardiamo tutti i clienti che volevano Accumulo: vedete come questo grosso flusso si divide alla fine, e quanti ne perdiamo a causa delle regole bancarie?"). Sintetizza l'intera Pipeline X in un'unica immagine.