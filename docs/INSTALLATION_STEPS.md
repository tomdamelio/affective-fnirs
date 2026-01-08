# Pasos de Instalación

## Después de cambiar environment.yml

Si modificaste `environment.yml` (como el cambio de pysnirf2):

```powershell
# 1. Eliminar entorno anterior
micromamba env remove -n affective-fnirs -y

# 2. Crear entorno nuevo
micromamba create -f environment.yml

# 3. Instalar paquete local
micromamba run -n affective-fnirs pip install -e .

# 4. Verificar
micromamba run -n affective-fnirs python scripts/verify_imports.py
```

## Primera instalación

```powershell
# 1. Crear entorno
micromamba create -f environment.yml

# 2. Instalar paquete
micromamba run -n affective-fnirs pip install -e .
```

Listo. Ahora puedes trabajar.
