# Datos Crudos (Raw Data)

⚠️ **ZONA DE SOLO LECTURA** ⚠️

Esta carpeta contiene los datos crudos del experimento. **NUNCA** modifiques archivos en esta carpeta.

## Estructura esperada

Los datos deben seguir el estándar BIDS (Brain Imaging Data Structure):

```
data/raw/
├── dataset_description.json
├── participants.tsv
├── participants.json
└── sub-<label>/
    └── ses-<label>/
        └── nirs/
            ├── sub-<label>_ses-<label>_task-<label>_nirs.snirf
            └── sub-<label>_ses-<label>_task-<label>_channels.tsv
```

## Cómo obtener los datos

[INSTRUCCIONES PENDIENTES: Agregar información sobre cómo obtener/descargar los datos del experimento]

## Datos derivados

Todos los datos procesados deben guardarse en `data/derivatives/<pipeline_name>/`.
