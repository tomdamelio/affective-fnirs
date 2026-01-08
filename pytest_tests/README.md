# Tests Automatizados (pytest)

Esta carpeta contiene tests formales automatizados usando pytest.

## Ejecutar tests

```powershell
# Todos los tests
micromamba run -n affective-fnirs pytest pytest_tests/ -v

# Con cobertura
micromamba run -n affective-fnirs pytest pytest_tests/ --cov=src/affective_fnirs --cov-report=html

# Un test específico
micromamba run -n affective-fnirs pytest pytest_tests/test_config.py -v
```

## Convenciones

- Archivos: `test_*.py` o `*_test.py`
- Clases: `Test*`
- Funciones: `test_*`
- Usar assertions claras
- Cada módulo en `src/` debe tener su `test_*.py` correspondiente

## Estructura

```
pytest_tests/
├── test_config.py          # Tests de configuraciones
├── test_preprocessing.py   # Tests de preprocesamiento (futuro)
├── test_io.py             # Tests de I/O (futuro)
└── test_analysis.py       # Tests de análisis (futuro)
```

## Diferencia con tests/

- **`tests/`**: Scripts sueltos, experimentales, no automatizados
- **`pytest_tests/`**: Tests formales, automatizados, con assertions
