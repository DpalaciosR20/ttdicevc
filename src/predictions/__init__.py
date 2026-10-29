"""Formato de predicciones por muestra (F1.5, docs/especificacion_predicciones.md)."""

__all__ = ['PredictionFormatError', 'load_predictions', 'validate_predictions']


def __getattr__(name):
    # Importación diferida: así `python -m src.predictions.validate` no carga el módulo dos veces
    if name in __all__:
        from src.predictions import validate
        return getattr(validate, name)
    raise AttributeError(name)
