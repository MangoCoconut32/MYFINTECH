# 📊 Sintesi Esecutiva: Recommender System Finanziario (Pipeline X)

## 🎯 Il Progetto in 10 Secondi (TL;DR)
Abbiamo costruito un motore di intelligenza artificiale per suggerire il prodotto finanziario perfetto per 5.000 clienti. Il risultato?
1. **Precisione da record:** Abbiamo superato un blocco storico, migliorando la precisione del **+5%** sul nostro target più difficile.
2. **100% Trasparenza (Zero "Scatole Nere"):** Il sistema è totalmente conforme alla direttiva MIFID. Spiega esattamente *perché* consiglia un prodotto, senza usare approssimazioni.
3. **Immediatamente Operativo:** Fornisce ai consulenti in filiale indicazioni chiare, inclusi consigli su come un cliente può diventare idoneo a un'offerta (es. "Aumenta l'educazione finanziaria").

---

## 🧱 I Due Muri (Il Problema di Partenza)
Assegnare il prodotto giusto a un cliente sembra facile, ma l'Intelligenza Artificiale in finanza si scontra con due ostacoli che non esistono in altri settori:
* **Il Muro della Precisione:** Capire chi vuole un prodotto "Income" (Rendita) è difficilissimo. I vecchi modelli si erano incagliati a un'accuratezza del 76%. Non riuscivamo ad andare oltre.
* **Il Muro della Compliance (MIFID II):** Le banche non possono usare algoritmi "Black Box" (Scatole Nere) come le classiche reti neurali, perché non sanno spiegare il motivo di una decisione. Se arriva un auditor, devi avere la formula matematica esatta, non un "il computer ha detto così".

---

## 🚀 La Soluzione: La "Pipeline X"
Invece di cercare un unico modello "magico" che facesse tutto a malapena, abbiamo ridisegnato l'intera architettura dai dati fino all'algoritmo.

### 1. Dati a prova di bomba (Zero truffe)
Abbiamo "congelato" i dati. In molti progetti AI, i modelli a volte "sbirciano" per sbaglio i dati di test, dando risultati gonfiati e finti (Data Leakage). Noi abbiamo creato un protocollo matematico che rende questo errore letteralmente impossibile. Quello che il modello promette nei test, lo mantiene nel mondo reale.

### 2. Più carburante per il motore: EDA e Deep Feature Synthesis
I 7 dati di base del cliente (Età, Reddito, ecc.) non bastavano. Abbiamo diviso il potenziamento dei dati (Data Engineering) in due fasi strategiche:
1. **Fase EDA (Domain Knowledge):** Guidati dal buon senso finanziario, abbiamo creato variabili "umane" come il *Reddito rapportato al Patrimonio* o la *Ricchezza pro-capite familiare*. Queste variabili rivelano il ciclo di vita effettivo del cliente che i numeri crudi nascondevano.
2. **Fase DFS (Forza Bruta):** Abbiamo poi accolto il lavoro del secondo sviluppatore, che tramite *Deep Feature Synthesis (DFS)* ha incrociato matematicamente ogni variabile esistente. 

**Perché 30 variabili? Il gap strutturale tra Alberi e Reti Neurali**
Espandere il dataset a 30 feature ha ottimizzato l'ambiente analitico per i Modelli ad Albero (Random Forest, XGBoost, EBM). Questi algoritmi esprimono la massima efficienza quando disattivano interazioni matematiche pre-calcolate, riducendo la necessità di generare ramificazioni profonde e instabili per identificare pattern logici complessi.

Di contro, l'alta collinearità introdotta da queste 30 feature ha causato deviazioni strutturali per le Reti Neurali (ANN). Nello specifico, il meccanismo di Attenzione Sparsa di TabNet si frammentava nell'analisi di metriche ridondanti, portando il modello in forte overfitting. Per ovviare a questo vincolo di architettura, abbiamo isolato per TabNet una "Vista Ibrida" rigorosamente potata a **15 feature totalmente decorrelate**, ripristinando la sua capacità predittiva.

### 3. I Due Motori Specializzati
Invece di un solo modello per tutto, abbiamo schierato due specialisti:

* **Per l'Accumulo ➔ La Scatola di Vetro (EBM):**
  Un modello potentissimo ma trasparente al 100%. L'EBM non è una scatola nera; ti fornisce la formula esatta della sua decisione. Dice: *"Ho abbassato il punteggio del 10% esclusivamente a causa dell'età"*. Addio stime approssimative. La Compliance dorme sonni tranquilli.
  
* **Per il Reddito (Income) ➔ Il Cecchino Deep Learning (TabNet V3 "Precision Strike"):**
  Per abbattere il famoso muro del 76%, abbiamo usato TabNet, una Rete Neurale avanzata nata dalla ricerca accademica SOTA. A differenza delle reti neurali tradizionali — che sono scatole nere al 100% — TabNet usa un meccanismo di **Attenzione Sparsa**: ad ogni decisione, rilascia lo scontrino esatto di quali variabili ha guardato di più per quel cliente. Grazie all'ingegnerizzazione di una vista "Ibrida" a **15 variabili strettamente decorrelate**, la rete neural focalizza l'attenzione chirurgicamente senza distrarsi su artefatti ridondanti. **Il risultato? Ha frantumato il muro del 76%, raggiungendo un solido 81.2% (ROC-AUC) in ambiente di test blindato.**

---

## 💼 Dal Numero alla Vendita (Impatto sul Business)
Un'alta percentuale di accuratezza non serve a niente se non fa vendere i prodotti giusti. Il nostro ultimo step ("Production Engine") prende le percentuali dei modelli e le trasforma in azioni per la banca:

1. **Filtro di Rischio (MIFID):** Il motore incrocia le probabilità con il profilo di rischio del cliente. Garantisce che non venga **mai** proposto un prodotto più rischioso di quanto il cliente possa sopportare.
2. **Nessuna Discriminazione (Fairness Audit):** Il sistema genera un report automatico che certifica alla direzione che l'algoritmo non ha "bias". Funziona bene in modo identico sia per i giovani che per gli anziani, sia per gli uomini che per le donne.
3. **Consigli per i Consulenti (Counterfactuals):** Se un cliente viene rifiutato dal modello per un prodotto interessante, il sistema dice al consulente bancario cosa manca: *"Se l'educazione finanziaria del cliente passasse da B ad A, il prodotto verrebbe sbloccato"*. Trasformiamo un rifiuto in un'opportunità di dialogo e consulenza.