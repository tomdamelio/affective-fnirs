# Scripts de Prueba

Esta carpeta contiene scripts sueltos para pruebas exploratorias, debugging y experimentación.

**NO** son tests formales con pytest. Son scripts que ejecutas manualmente para probar cosas.

## Uso

```powershell
# Ejecutar cualquier script de prueba
micromamba run -n affective-fnirs python tests/mi_script_de_prueba.py
```

## Diferencia con pytest_tests/

- **`tests/`**: Scripts sueltos, experimentales, no automatizados
- **`pytest_tests/`**: Tests formales, automatizados, con assertions

## Ejemplos de lo que va aquí

- Scripts para explorar datos
- Pruebas rápidas de funciones
- Debugging de problemas específicos
- Prototipos de análisis
- Visualizaciones exploratorias

No necesitas seguir ninguna convención de nombres. Usa nombres descriptivos como:
- `prueba_cargar_snirf.py`
- `debug_montaje.py`
- `explorar_canales.py`
- `test_rapido_filtrado.py`
