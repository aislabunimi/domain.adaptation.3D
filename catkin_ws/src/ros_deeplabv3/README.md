# Ros Deeplab V3

Questo pacchetto contiene due nodi principali:

- `deeplab_segmenter`: esegue segmentazione semantica in tempo reale su immagini.
- `deeplab_finetune_service`: espone un servizio ROS per effettuare il fine-tuning del modello DeepLabV3 con un dataset personalizzato.

---

## `deeplab_segmenter`

### Funzione
Segmenta semanticamente immagini ricevute in tempo reale utilizzando un modello DeepLabV3 pre-addestrato (o fine-tunato).

### Input
- **Topic:** `/camera/image_raw`
- **Tipo:** `sensor_msgs/Image`
- **Descrizione:** Immagine RGB.

### Output
- **Topic:** `/deeplab/segmented_image`
- **Tipo:** `sensor_msgs/Image` (mono8)
- **Descrizione:** Mappa di segmentazione semanticamente etichettata.

---

## `deeplab_finetune_service`

### Funzione
Fornisce un servizio ROS che permette di fine-tunare il modello DeepLabV3 su un nuovo dataset costituito da immagini e maschere di segmentazione.

### Servizio
- **Nome:** `/deeplab_finetune`
- **Tipo:** `Finetune.srv`

#### Struttura `Finetune.srv`
```srv
string dataset_path        # Directory con le immagini (RGB)
uint32 num_epochs          # Numero di epoche per il fine-tuning
uint32 num_classes         # Numero di classi semanticamente etichettate
---
bool success               # Esito del fine-tuning
string message             # Messaggio di ritorno
